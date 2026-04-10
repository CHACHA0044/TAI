---
title: TruthGuard AI Backend
emoji: 🛡️
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
---

# TruthGuard AI — All-in-One Backend

This container runs the FastAPI server, Celery Worker, and Redis storage for the TruthGuard platform.

## Architecture 
- **API**: FastAPI (Port 7860)
- **Worker**: Celery (Concurrency: 1)
- **Broker**: Redis (Internal)

## Deployment Note
This space is optimized for the **16GB RAM Free Tier** provided by Hugging Face Spaces. It uses a single container orchestration to keep costs at zero.
