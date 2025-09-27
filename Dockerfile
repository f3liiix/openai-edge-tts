FROM python:3.12-slim

ARG INSTALL_FFMPEG=true
WORKDIR /app

# Install ffmpeg by default (can disable via build-arg)
RUN if [ "$INSTALL_FFMPEG" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy requirements and install them
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the app directory
COPY app/ /app

# Command to run the server
CMD ["python", "/app/server.py"]
