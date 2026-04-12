# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    redis-server \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the backend folder and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the backend folder to the working directory
COPY backend .

# Pre-cache models to avoid runtime 429 rate limits
RUN python utils/cache_models.py

# Make start script executable
RUN chmod +x start.sh

# HF Spaces required port
EXPOSE 7860

# Set Hugging Face cache and memory optimisation flags
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use the same port as HF (7860) for everything
ENV PORT=7860

# Start all services via start.sh
CMD ["./start.sh"]
