#!/bin/bash
set -e

# --- Redis ---
echo "Starting Redis server..."
redis-server --daemonize yes --maxmemory 512mb --maxmemory-policy allkeys-lru

until redis-cli ping 2>/dev/null | grep -q PONG; do
  echo "Waiting for Redis..."
  sleep 1
done
echo "Redis is ready!"

# --- Celery worker ---
echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info --concurrency=1 &

# --- FastAPI / Uvicorn ---
echo "Starting FastAPI on port 7860..."
exec uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
