#!/bin/bash

# Start Redis in the background
echo "Starting Redis server..."
redis-server --daemonize yes

# Wait for Redis to be ready
until redis-cli ping | grep -q PONG; do
  echo "Waiting for Redis..."
  sleep 1
done
echo "Redis is ready!"

# Start Celery Worker in the background
echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info --concurrency=1 &

# Start FastAPI/Uvicorn in the foreground
echo "Starting FastAPI on port 7860..."
uvicorn main:app --host 0.0.0.0 --port 7860
