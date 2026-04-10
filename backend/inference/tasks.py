import logging
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
def process_video_task(self, video_bytes: bytes, filename: str, content_hash: str):
    logger.info(f"Starting video analysis task for {filename} (Hash: {content_hash})")
    
    # 1. Double check cache just in case of parallel distinct submits
    cached = get_cached_result("video", content_hash)
    if cached:
        logger.info("Found in cache during task startup.")
        return cached

    # 2. Process
    try:
        # We need to update task state during processing if we want progressive UI
        self.update_state(state="PROCESSING", meta={"step": "Extracting frames and scanning neural artifacts..."})
        
        engine = get_engine()
        result = engine.analyze(video_bytes, filename=filename)
        
        # 3. Cache the successful result
        set_cached_result("video", content_hash, result)
        
        return result
    except Exception as e:
        logger.exception(f"Video analysis task failed: {e}")
        # Celery marks it as FAILURE if exception is raised
        raise e
