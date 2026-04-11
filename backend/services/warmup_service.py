"""
Warmup service — keeps the HF Spaces container from sleeping.

Runs as a background daemon thread.  On each tick it:
  1. Sends a GET /health request to the local FastAPI server.
  2. Executes a batch of random math operations (keeps CPU slightly active).
  3. Logs the result so the HF Spaces log stream stays live.

Start it by calling ``start_warmup_service()`` once during app initialization.
"""

import logging
import math
import os
import random
import threading
import time

import requests

logger = logging.getLogger("truthguard.warmup")

_POLL_INTERVAL_S = int(os.getenv("WARMUP_INTERVAL_S", "45"))
_LOCAL_PORT = int(os.getenv("PORT", "7860"))
_HEALTH_URL = f"http://localhost:{_LOCAL_PORT}/health"


def _math_workload() -> float:
    """Perform random floating-point calculations to keep the CPU warm."""
    total = 0.0
    ops = random.randint(5_000, 20_000)
    for _ in range(ops):
        x = random.uniform(0.001, 1_000.0)
        total += math.sqrt(x) * math.log(x + 1) * math.sin(x)
    return total


def _run_loop() -> None:
    tick = 0
    # Give the main server time to start before the first poll
    time.sleep(15)
    while True:
        tick += 1
        try:
            resp = requests.get(_HEALTH_URL, timeout=10)
            status = resp.json().get("status", "unknown") if resp.ok else f"HTTP {resp.status_code}"
        except Exception as exc:
            status = f"error ({exc})"

        result = _math_workload()
        logger.info(
            f"[warmup tick={tick}] health={status} | "
            f"math_result={result:.4f} | next_in={_POLL_INTERVAL_S}s"
        )
        time.sleep(_POLL_INTERVAL_S)


_WARMUP_ENABLED = os.getenv("WARMUP_ENABLED", "true").lower() in ("1", "true", "yes")

# Singleton guard — prevents duplicate threads if called more than once.
_started = False
_lock = threading.Lock()


def start_warmup_service() -> threading.Thread | None:
    """
    Spawn the warmup loop as a background daemon thread and return it.
    Safe to call multiple times — subsequent calls are no-ops.
    Gated by the ``WARMUP_ENABLED`` environment variable (default: true).
    """
    global _started
    if not _WARMUP_ENABLED:
        logger.info("Warmup service disabled (WARMUP_ENABLED != true)")
        return None
    with _lock:
        if _started:
            logger.info("Warmup service already running — skipping duplicate start")
            return None
        _started = True
    thread = threading.Thread(target=_run_loop, name="warmup-service", daemon=True)
    thread.start()
    logger.info(f"Warmup service started (interval={_POLL_INTERVAL_S}s, port={_LOCAL_PORT})")
    return thread
