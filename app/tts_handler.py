# tts_handler.py

import edge_tts
import asyncio
import tempfile
import subprocess
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TICKS_PER_SECOND = 10_000_000  # Azure edge-tts reports offsets/durations in 100ns units
DEFAULT_SEGMENT_MAX_GAP = float(os.getenv('SUBTITLE_MAX_GAP', '0.4'))
DEFAULT_TAIL_SILENCE_DURATION = float(os.getenv('AUDIO_TAIL_SILENCE_DURATION', '0.3'))
DEFAULT_TAIL_SILENCE_THRESHOLD_DB = float(os.getenv('AUDIO_TAIL_SILENCE_THRESHOLD_DB', '-50'))

from shutil import which

from utils import DETAILED_ERROR_LOGGING
from config import DEFAULT_CONFIGS

# Language default (environment variable)
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])

# OpenAI voice names mapped to edge-tts equivalents
voice_mapping = {
    'alloy': 'en-US-JennyNeural',
    'ash': 'en-US-AndrewNeural',
    'ballad': 'en-GB-ThomasNeural',
    'coral': 'en-AU-NatashaNeural',
    'echo': 'en-US-GuyNeural',
    'fable': 'en-GB-SoniaNeural',
    'nova': 'en-US-AriaNeural',
    'onyx': 'en-US-EricNeural',
    'sage': 'en-US-JennyNeural',
    'shimmer': 'en-US-EmmaNeural',
    'verse': 'en-US-BrianNeural',
}

model_data = [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"},
        {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"}
    ]

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    return which('ffmpeg') is not None

async def _generate_audio_stream(text, voice, speed):
    """Generate streaming TTS audio using edge-tts."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is
    
    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"
    
    # Create the communicator for streaming
    communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
    
    # Stream the audio data
    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

def generate_speech_stream(text, voice, speed=1.0):
    """Generate streaming speech audio (synchronous wrapper)."""
    return asyncio.run(_generate_audio_stream(text, voice, speed))

def _coerce_ticks(value: Optional[int]) -> int:
    if value is None:
        return 0
    return int(value)


def _ticks_to_seconds(value: int) -> float:
    return value / TICKS_PER_SECOND


def _should_end_segment(word_text: str, previous_end: Optional[float], current_start: float,
                        max_gap: float) -> bool:
    if not word_text:
        return False

    # Break on explicit punctuation or newline characters
    if any(ch in word_text for ch in ("\n", "\r")):
        return True
    if word_text[-1] in ".!?！？。;；":
        return True

    if previous_end is None:
        return False

    # If there is a long gap between words, start a new subtitle segment
    return (current_start - previous_end) > max_gap


def _append_word_text(buffer: str, word_text: str) -> str:
    if not buffer:
        return word_text

    if not word_text:
        return buffer

    prev_char = buffer[-1]
    next_char = word_text[0]

    if prev_char.isspace():
        return buffer + word_text

    # Add a space between contiguous latin words to improve readability
    if prev_char.isalnum() and next_char.isalnum() and prev_char.isascii() and next_char.isascii():
        return f"{buffer} {word_text}"

    return buffer + word_text


def _segments_from_word_boundaries(word_boundaries: List[Dict[str, float]],
                                   max_gap: float) -> List[Dict[str, float]]:
    segments: List[Dict[str, float]] = []
    current_text = ""
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for boundary in word_boundaries:
        word_text = boundary["text"]
        word_start = boundary["start"]
        word_end = boundary["end"]

        if current_start is None:
            current_start = word_start
            current_end = word_end
            current_text = word_text
            continue

        # Decide whether to close the current segment before appending this word
        should_close = _should_end_segment(word_text, current_end, word_start, max_gap)

        if should_close:
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": current_end if current_end is not None else word_start
                })
            # Reset for the next segment
            current_start = word_start
            current_end = word_end
            current_text = word_text
            continue

        current_text = _append_word_text(current_text, word_text)
        current_end = word_end

    if current_text.strip() and current_start is not None and current_end is not None:
        segments.append({
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end
        })

    return segments


def _trim_trailing_silence(audio_path: str, response_format: str,
                           stop_duration: float = DEFAULT_TAIL_SILENCE_DURATION,
                           stop_threshold_db: float = DEFAULT_TAIL_SILENCE_THRESHOLD_DB) -> None:
    """Use ffmpeg to remove trailing silence from an audio file in-place."""
    if stop_duration <= 0:
        return

    if not is_ffmpeg_installed():
        if DETAILED_ERROR_LOGGING:
            print("[trim] ffmpeg not available; skipping tail silence removal")
        return

    suffix = Path(audio_path).suffix or f".{response_format}"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    trimmed_path = tmp_file.name
    tmp_file.close()

    codec_map = {
        'mp3': ['-codec:a', 'libmp3lame', '-b:a', '192k'],
        'aac': ['-codec:a', 'aac', '-b:a', '192k'],
        'opus': ['-codec:a', 'libopus'],
        'flac': ['-codec:a', 'flac'],
        'wav': ['-codec:a', 'pcm_s16le'],
    }

    codec_args = codec_map.get(response_format, ['-codec:a', 'copy'])
    # Using a filter requires re-encoding; fall back to PCM if codec copy is selected
    if codec_args == ['-codec:a', 'copy']:
        codec_args = ['-codec:a', 'pcm_s16le']

    silence_filter = (
        f"silenceremove=start_periods=0:stop_periods=1:"
        f"stop_duration={stop_duration}:stop_threshold={stop_threshold_db}dB"
    )

    command = [
        'ffmpeg',
        '-y',
        '-i', audio_path,
        '-af', silence_filter,
        *codec_args,
        trimmed_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if DETAILED_ERROR_LOGGING:
            print(f"[trim] Removed trailing silence from {audio_path}")
    except subprocess.CalledProcessError as exc:
        Path(trimmed_path).unlink(missing_ok=True)
        if DETAILED_ERROR_LOGGING:
            print(f"[trim] ffmpeg failed for {audio_path}: {exc}")
        return

    try:
        Path(trimmed_path).replace(audio_path)
    except OSError as exc:
        # If replacing fails, leave the original file untouched
        Path(trimmed_path).unlink(missing_ok=True)
        if DETAILED_ERROR_LOGGING:
            print(f"[trim] Failed to replace original audio after trimming {audio_path}: {exc}")


def _format_srt_timestamp(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    if milliseconds == 1000:
        milliseconds = 0
        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _format_vtt_timestamp(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    if milliseconds == 1000:
        milliseconds = 0
        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def _subtitle_from_segments(segments: List[Dict[str, float]], subtitle_format: Optional[str]) -> Optional[str]:
    if not subtitle_format or not segments:
        return None

    subtitle_format = subtitle_format.lower()

    if subtitle_format not in {"srt", "vtt", "webvtt"}:
        raise ValueError(f"Unsupported subtitle format '{subtitle_format}'")

    if subtitle_format == "srt":
        lines: List[str] = []
        for idx, item in enumerate(segments, start=1):
            lines.append(str(idx))
            lines.append(f"{_format_srt_timestamp(item['start'])} --> {_format_srt_timestamp(item['end'])}")
            lines.append(item['text'])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    # Default to WebVTT style output
    lines = ["WEBVTT", ""]
    for item in segments:
        lines.append(f"{_format_vtt_timestamp(item['start'])} --> {_format_vtt_timestamp(item['end'])}")
        lines.append(item['text'])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_audio_result(audio_path: str,
                        include_word_boundaries: bool,
                        subtitle_format: Optional[str],
                        word_boundary_payloads: List[Dict[str, object]],
                        segment_max_gap: float) -> Dict[str, object]:
    metadata = None
    segments: Optional[List[Dict[str, float]]] = None
    subtitle_payload = None

    if include_word_boundaries and word_boundary_payloads:
        metadata = _normalize_word_boundaries(word_boundary_payloads)
        segments = _segments_from_word_boundaries(metadata, segment_max_gap)
    if subtitle_format and segments:
        subtitle_payload = _subtitle_from_segments(segments, subtitle_format)

    return {
        "audio_path": audio_path,
        "word_boundaries": metadata,
        "subtitle_format": subtitle_format if subtitle_payload else None,
        "subtitle": subtitle_payload,
        "segments": segments,
    }


def _normalize_word_boundaries(boundaries: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
    for boundary in boundaries:
        text = str(boundary.get("text", ""))
        offset_ticks = _coerce_ticks(boundary.get("offset") or boundary.get("Offset"))
        duration_ticks = _coerce_ticks(boundary.get("duration") or boundary.get("Duration"))
        start_seconds = _ticks_to_seconds(offset_ticks)
        end_seconds = _ticks_to_seconds(offset_ticks + duration_ticks)

        normalized.append({
            "text": text,
            "offset_ticks": offset_ticks,
            "duration_ticks": duration_ticks,
            "start": start_seconds,
            "end": end_seconds,
        })

    return normalized


async def _generate_audio(text, voice, response_format, speed, include_word_boundaries=False,
                          subtitle_format: Optional[str] = None,
                          segment_max_gap: float = DEFAULT_SEGMENT_MAX_GAP):
    """Generate TTS audio and optionally collect metadata and convert to different formats."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is

    # Generate the TTS output in mp3 format first
    temp_mp3_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3_file_obj.name

    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    boundary_mode = "WordBoundary" if include_word_boundaries else "SentenceBoundary"

    communicator = edge_tts.Communicate(
        text=text,
        voice=edge_tts_voice,
        rate=speed_rate,
        boundary=boundary_mode,
    )

    word_boundary_payloads: List[Dict[str, object]] = []

    with open(temp_mp3_path, "wb") as audio_file:
        async for chunk in communicator.stream():
            chunk_type = chunk.get("type") if isinstance(chunk, dict) else None
            if chunk_type == "audio":
                audio_file.write(chunk.get("data", b""))
            elif chunk_type == "WordBoundary" and include_word_boundaries:
                word_boundary_payloads.append(chunk)

    temp_mp3_file_obj.close() # Explicitly close our file object for the initial mp3

    # If the requested format is mp3, return the generated file directly
    if response_format == "mp3":
        _trim_trailing_silence(
            temp_mp3_path,
            response_format,
        )
        return _build_audio_result(
            temp_mp3_path,
            include_word_boundaries,
            subtitle_format,
            word_boundary_payloads,
            segment_max_gap,
        )

    # Check if FFmpeg is installed
    if not is_ffmpeg_installed():
        print("FFmpeg is not available. Returning unmodified mp3 file.")
        _trim_trailing_silence(
            temp_mp3_path,
            response_format,
        )
        return _build_audio_result(
            temp_mp3_path,
            include_word_boundaries,
            subtitle_format,
            word_boundary_payloads,
            segment_max_gap,
        )

    # Create a new temporary file for the converted output
    converted_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    converted_path = converted_file_obj.name
    converted_file_obj.close() # Close file object, ffmpeg will write to the path

    # Build the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", temp_mp3_path,  # Input file path
        "-c:a", {
            "aac": "aac",
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "opus": "libopus",
            "flac": "flac"
        }.get(response_format, "aac"),  # Default to AAC if unknown
    ]

    if response_format != "wav":
        ffmpeg_command.extend(["-b:a", "192k"])

    ffmpeg_command.extend([
        "-f", {
            "aac": "mp4",  # AAC in MP4 container
            "mp3": "mp3",
            "wav": "wav",
            "opus": "ogg",
            "flac": "flac"
        }.get(response_format, response_format),  # Default to matching format
        "-y",  # Overwrite without prompt
        converted_path  # Output file path
    ])

    try:
        # Run FFmpeg command and ensure no errors occur
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Clean up potentially created (but incomplete) converted file
        Path(converted_path).unlink(missing_ok=True)
        # Clean up the original mp3 file as well, since conversion failed
        Path(temp_mp3_path).unlink(missing_ok=True)
        
        if DETAILED_ERROR_LOGGING:
            error_message = f"FFmpeg error during audio conversion. Command: '{' '.join(e.cmd)}'. Stderr: {e.stderr.decode('utf-8', 'ignore')}"
            print(error_message) # Log for server-side diagnosis
        else:
            error_message = f"FFmpeg error during audio conversion: {e}"
            print(error_message) # Log a simpler message
        raise RuntimeError(f"FFmpeg error during audio conversion: {e}") # The raised error will still have details via e

    # Clean up the original temporary file (original mp3) as it's now converted
    Path(temp_mp3_path).unlink(missing_ok=True)

    _trim_trailing_silence(
        converted_path,
        response_format,
    )

    return _build_audio_result(
        converted_path,
        include_word_boundaries,
        subtitle_format,
        word_boundary_payloads,
        segment_max_gap,
    )


def generate_speech(text, voice, response_format, speed=1.0, *, include_word_boundaries: bool = False,
                    subtitle_format: Optional[str] = None,
                    segment_max_gap: Optional[float] = None):
    segment_max_gap = segment_max_gap if segment_max_gap is not None else DEFAULT_SEGMENT_MAX_GAP
    result = asyncio.run(
        _generate_audio(
            text,
            voice,
            response_format,
            speed,
            include_word_boundaries=include_word_boundaries or bool(subtitle_format),
            subtitle_format=subtitle_format,
            segment_max_gap=segment_max_gap,
        )
    )

    if isinstance(result, str):
        # Backwards compatibility: when _generate_audio returned a path (mp3 without metadata)
        return result

    if not include_word_boundaries and not subtitle_format:
        return result["audio_path"]

    return result

def get_models():
    return model_data

def get_models_formatted():
    return [{ "id": x["id"] } for x in model_data]

def get_voices_formatted():
    return [{ "id": k, "name": v } for k, v in voice_mapping.items()]

async def _get_voices(language=None):
    # List all voices, filter by language if specified
    all_voices = await edge_tts.list_voices()
    language = language or DEFAULT_LANGUAGE  # Use default if no language specified
    filtered_voices = [
        {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
        for v in all_voices if language == 'all' or language is None or v['Locale'] == language
    ]
    return filtered_voices

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    Converts a multiplicative speed value to the edge-tts "rate" format.
    
    Args:
        speed (float): The multiplicative speed value (e.g., 1.5 for +50%, 0.5 for -50%).
    
    Returns:
        str: The formatted "rate" string (e.g., "+50%" or "-50%").
    """
    if speed < 0 or speed > 2:
        raise ValueError("Speed must be between 0 and 2 (inclusive).")

    # Convert speed to percentage change
    percentage_change = (speed - 1) * 100

    # Format with a leading "+" or "-" as required
    return f"{percentage_change:+.0f}%"
