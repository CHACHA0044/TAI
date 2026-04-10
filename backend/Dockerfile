# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV requires libgl1 and libglib2.0)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend directory
COPY . .

# Default port (FastAPI)
EXPOSE 8000

# Set Hugging Face cache to point to a mounted volume inside docker
ENV HF_HOME=/app/.cache/huggingface

# Start FastAPI server by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
