# server.py

from flask import Flask, request, send_file, jsonify, Response
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
import os
import traceback
import json
import base64

from config import DEFAULT_CONFIGS
from handle_text import prepare_tts_input_with_context
from tts_handler import generate_speech, generate_speech_stream, get_models_formatted, get_voices, get_voices_formatted
from utils import getenv_bool, require_api_key, AUDIO_FORMAT_MIME_TYPES, DETAILED_ERROR_LOGGING

app = Flask(__name__)
load_dotenv()

API_KEY = os.getenv('API_KEY', DEFAULT_CONFIGS["API_KEY"])
PORT = int(os.getenv('PORT', str(DEFAULT_CONFIGS["PORT"])))

DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', DEFAULT_CONFIGS["DEFAULT_VOICE"])
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', DEFAULT_CONFIGS["DEFAULT_RESPONSE_FORMAT"])
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', str(DEFAULT_CONFIGS["DEFAULT_SPEED"])))

REMOVE_FILTER = getenv_bool('REMOVE_FILTER', DEFAULT_CONFIGS["REMOVE_FILTER"])
EXPAND_API = getenv_bool('EXPAND_API', DEFAULT_CONFIGS["EXPAND_API"])

# DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'tts-1')

# Currently in "beta" — needs more extensive testing where drop-in replacement warranted
def generate_sse_audio_stream(text, voice, speed):
    """Generator function for SSE streaming with JSON events."""
    try:
        # Generate streaming audio chunks and convert to SSE format
        for chunk in generate_speech_stream(text, voice, speed):
            # Base64 encode the audio chunk
            encoded_audio = base64.b64encode(chunk).decode('utf-8')
            
            # Create SSE event for audio delta
            event_data = {
                "type": "speech.audio.delta",
                "audio": encoded_audio
            }
            
            # Format as SSE event
            yield f"data: {json.dumps(event_data)}\n\n"
        
        # Send completion event
        completion_event = {
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": len(text.split()),  # Rough estimate
                "output_tokens": 0,  # Edge TTS doesn't provide this
                "total_tokens": len(text.split())
            }
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
        
    except Exception as e:
        print(f"Error during SSE streaming: {e}")
        # Send error event
        error_event = {
            "type": "error",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"

# OpenAI endpoint format
@app.route('/v1/audio/speech', methods=['POST'])
@app.route('/audio/speech', methods=['POST'])  # Add this line for the alias
@require_api_key
def text_to_speech():
    try:
        data = request.json
        if not data or 'input' not in data:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        text = data.get('input')
        if text is None:
            return jsonify({"error": "'input' must not be null"}), 400
        if isinstance(text, str):
            if not text.strip():
                return jsonify({"error": "'input' must not be empty"}), 400
        else:
            # Could support SSML or arrays in future; for now only plain string
            return jsonify({"error": "'input' must be a string"}), 400

        if not REMOVE_FILTER:
            text = prepare_tts_input_with_context(text)
            if not text:
                return jsonify({"error": "Text became empty after cleaning; nothing to synthesize"}), 400

        # model = data.get('model', DEFAULT_MODEL)
        voice = data.get('voice', DEFAULT_VOICE)
        response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
        try:
            speed = float(data.get('speed', DEFAULT_SPEED))
        except (TypeError, ValueError):
            return jsonify({"error": "'speed' must be a number"}), 400
        if not (0.0 <= speed <= 2.0):
            return jsonify({"error": "'speed' must be between 0 and 2"}), 400

        include_word_boundaries = bool(data.get('include_word_boundaries', False))
        subtitle_format = data.get('subtitle_format')
        if subtitle_format:
            subtitle_format = subtitle_format.strip().lower()

        segment_max_gap = data.get('segment_max_gap')

        try:
            segment_max_gap = float(segment_max_gap) if segment_max_gap is not None else None
        except (TypeError, ValueError):
            return jsonify({"error": "segment_max_gap must be a number"}), 400

        response_mode = data.get('response_mode', 'binary')
        if isinstance(response_mode, str):
            response_mode = response_mode.lower()
        else:
            response_mode = 'binary'

        metadata_requested = include_word_boundaries or bool(subtitle_format) or bool(data.get('return_metadata', False)) or response_mode == 'json'
        
        # Check stream format - only "sse" triggers streaming
        stream_format = data.get('stream_format', 'audio')  # 'audio' (default) or 'sse'
        
        mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format, "audio/mpeg")
        
        if stream_format == 'sse':
            # Return SSE streaming response with JSON events
            def generate_sse():
                for event in generate_sse_audio_stream(text, voice, speed):
                    yield event
            
            return Response(
                generate_sse(),
                mimetype='text/event-stream',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'  # Disable nginx buffering
                }
            )
        else:
            # Return raw audio data (like OpenAI) or JSON payload when metadata requested
            try:
                synthesis_result = generate_speech(
                    text,
                    voice,
                    response_format,
                    speed,
                    include_word_boundaries=include_word_boundaries or bool(subtitle_format),
                    subtitle_format=subtitle_format,
                    segment_max_gap=segment_max_gap,
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                # Provide clearer mapping for common upstream issues
                message = str(exc)
                if "No audio was received" in message:
                    return jsonify({
                        "error": "No audio received from TTS upstream",
                        "hint": "Try a different voice, simplify text, or reduce request rate",
                    }), 502
                raise

            if isinstance(synthesis_result, dict):
                output_file_path = synthesis_result["audio_path"]
                metadata_payload = synthesis_result
            else:
                output_file_path = synthesis_result
                metadata_payload = None

            # Read the file and return raw audio data
            with open(output_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Clean up the temporary file
            try:
                os.unlink(output_file_path)
            except OSError:
                pass  # File might already be cleaned up

            if not metadata_requested:
                return Response(
                    audio_data,
                    mimetype=mime_type,
                    headers={
                        'Content-Type': mime_type,
                        'Content-Length': str(len(audio_data))
                    }
                )

            encoded_audio = base64.b64encode(audio_data).decode('utf-8')

            response_payload = {
                "audio": encoded_audio,
                "audio_format": response_format,
                "mime_type": mime_type,
                "size_bytes": len(audio_data),
            }

            if metadata_payload:
                word_boundaries = metadata_payload.get("word_boundaries")
                if word_boundaries:
                    response_payload["word_boundaries"] = word_boundaries

                segments = metadata_payload.get("segments")
                if segments:
                    response_payload["segments"] = segments

                subtitle_text = metadata_payload.get("subtitle")
                if subtitle_text:
                    response_payload["subtitle"] = subtitle_text
                    response_payload["subtitle_format"] = metadata_payload.get("subtitle_format") or subtitle_format

                response_payload["audio_duration_seconds"] = metadata_payload.get("audio_duration")

            return jsonify(response_payload)
            
    except Exception as e:
        if DETAILED_ERROR_LOGGING:
            app.logger.error(f"Error in text_to_speech: {str(e)}\n{traceback.format_exc()}")
        else:
            app.logger.error(f"Error in text_to_speech: {str(e)}")
        # Return a 500 error for unhandled exceptions, which is more standard than 400
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

# OpenAI endpoint format
@app.route('/v1/models', methods=['GET', 'POST'])
@app.route('/models', methods=['GET', 'POST'])
@app.route('/v1/audio/models', methods=['GET', 'POST'])
@app.route('/audio/models', methods=['GET', 'POST'])
def list_models():
    return jsonify({"models": get_models_formatted()})

# OpenAI endpoint format
@app.route('/v1/audio/voices', methods=['GET', 'POST'])
@app.route('/audio/voices', methods=['GET', 'POST'])
def list_voices_formatted():
    return jsonify({"voices": get_voices_formatted()})

@app.route('/v1/voices', methods=['GET', 'POST'])
@app.route('/voices', methods=['GET', 'POST'])
@require_api_key
def list_voices():
    specific_language = None

    data = request.args if request.method == 'GET' else request.json
    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET', 'POST'])
@app.route('/voices/all', methods=['GET', 'POST'])
@require_api_key
def list_all_voices():
    return jsonify({"voices": get_voices('all')})

"""
Support for ElevenLabs and Azure AI Speech
    (currently in beta)
"""

# http://localhost:5050/elevenlabs/v1/text-to-speech
# http://localhost:5050/elevenlabs/v1/text-to-speech/en-US-AndrewNeural
@app.route('/elevenlabs/v1/text-to-speech/<voice_id>', methods=['POST'])
@require_api_key
def elevenlabs_tts(voice_id):
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the incoming JSON payload
    try:
        payload = request.json
        if not payload or 'text' not in payload:
            return jsonify({"error": "Missing 'text' in request body"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400

    text = payload['text']

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    voice = voice_id  # ElevenLabs uses the voice_id in the URL

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED  # Optional customization via payload.get('speed', DEFAULT_SPEED)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

# tts.speech.microsoft.com/cognitiveservices/v1
# https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
# http://localhost:5050/azure/cognitiveservices/v1
@app.route('/azure/cognitiveservices/v1', methods=['POST'])
@require_api_key
def azure_tts():
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the SSML payload
    try:
        ssml_data = request.data.decode('utf-8')
        if not ssml_data:
            return jsonify({"error": "Missing SSML payload"}), 400

        # Extract the text and voice from SSML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(ssml_data)
        text = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').text
        voice = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').get('name')
    except Exception as e:
        return jsonify({"error": f"Invalid SSML payload: {str(e)}"}), 400

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

print(f" Edge TTS (Free Azure TTS) Replacement for OpenAI's TTS API")
print(f" ")
print(f" * Serving OpenAI Edge TTS")
print(f" * Server running on http://localhost:{PORT}")
print(f" * TTS Endpoint: http://localhost:{PORT}/v1/audio/speech")
print(f" ")

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
