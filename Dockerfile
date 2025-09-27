FROM python:3.12-slim

ARG INSTALL_FFMPEG=false
WORKDIR /app

# Install ffmpeg conditionally
RUN if [ "$INSTALL_FFMPEG" = "true" ]; then \
    apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*; \
    fi

# 在执行 pip install 之前设置环境变量
ENV HTTP_PROXY=http://172.18.0.1:7890
ENV HTTPS_PROXY=http://172.18.0.1:7890
ENV NO_PROXY=localhost,127.0.0.1,::1

# Copy requirements and install them
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the app directory
COPY app/ /app

# Command to run the server
CMD ["python", "/app/server.py"]
