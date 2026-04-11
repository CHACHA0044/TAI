import logging
import os
from celery_app import celery_app
from utils.cache import set_cached_result, get_cached_result

logger = logging.getLogger("truthguard.tasks")

# Lazy loading of engine to avoid loading it on every worker if not strictly needed immediately,
# but it will be loaded when the first task runs.
_video_engine = None

def get_engine():
    global _video_engine
    if _video_engine is None:
        from inference.video_engine import VideoEngine
        _video_engine = VideoEngine()
    return _video_engine

@celery_app.task(bind=True)
def process_video_task(self, video_path: str, filename: str, content_hash: str):
    """
    Celery task for async video analysis.

    Receives the *path* of the uploaded video (not the bytes themselves) to
    avoid JSON-serialisation failures and excessive Redis memory usage.
    The temp file is deleted after analysis completes (success or failure).
    """
    logger.info(f"Starting video analysis task for {filename} (Hash: {content_hash}, path: {video_path})")
    
    # 1. Double check cache just in case of parallel distinct submits
    cached = get_cached_result("video", content_hash)
    if cached:
        logger.info("Found in cache during task startup.")
        _cleanup(video_path)
        return cached

    # 2. Read video bytes from the temp file written by the API worker
    try:
        with open(video_path, "rb") as fh:
            video_bytes = fh.read()
    except FileNotFoundError:
        logger.error(f"Temp video file not found: {video_path}")
        raise

    # 3. Process
    try:
        self.update_state(state="PROCESSING", meta={"step": "Extracting frames and scanning neural artifacts..."})
        
        engine = get_engine()
        result = engine.analyze(video_bytes, filename=filename)
        
        # 4. Cache the successful result
        set_cached_result("video", content_hash, result)
        
        return result
    except Exception as e:
        logger.exception(f"Video analysis task failed: {e}")
        raise e
    finally:
        _cleanup(video_path)


def _cleanup(path: str) -> None:
    """Remove a temp file, swallowing errors so they don't mask task failures."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug(f"Cleaned up temp file: {path}")
    except OSError as exc:
        logger.warning(f"Could not remove temp file {path}: {exc}")
