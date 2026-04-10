# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# 1. OpenCV requires libgl1 and libglib2.0
# 2. Redis-server for internal task queue
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend directory
COPY . .

# Make start script executable
RUN chmod +x start.sh

# HF Spaces required port
EXPOSE 7860

# Set Hugging Face cache
ENV HF_HOME=/app/.cache/huggingface

# Start all services via start.sh
CMD ["./start.sh"]
