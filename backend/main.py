import logging
import os

# Hugging Face cache - prefer environment variable, default to /app cache
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/app/.cache/huggingface"
import sys
import time

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
from celery.result import AsyncResult

from utils.cache import get_cached_result, set_cached_result
from utils.hashing import get_content_hash, get_text_hash
from celery_app import celery_app
from inference.tasks import process_video_task

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("truthguard")

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(title="TruthGuard AI — Text Verification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    from services.warmup_service import start_warmup_service
    start_warmup_service()
    logger.info("Warmup service launched on startup")

# ------------------------------------------------------------------
# Lazy engine init (avoids import-time crashes if deps are missing)
# ------------------------------------------------------------------
_engine = None
_image_engine = None


def get_engine():
    global _engine
    if _engine is None:
        logger.info("Initializing InferenceEngine...")
        from inference.engine import InferenceEngine
        _engine = InferenceEngine()
        logger.info("InferenceEngine ready")
    return _engine


def get_image_engine():
    global _image_engine
    if _image_engine is None:
        logger.info("Initializing ImageEngine...")
        from inference.image_engine import ImageEngine
        _image_engine = ImageEngine()
        logger.info("ImageEngine ready")
    return _image_engine


_video_engine = None


def get_video_engine():
    global _video_engine
    if _video_engine is None:
        logger.info("Initializing VideoEngine...")
        from inference.video_engine import VideoEngine
        _video_engine = VideoEngine()
        logger.info("VideoEngine ready")
    return _video_engine


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    """Accepts 'content' (canonical), but also 'text' or 'url' for compatibility."""
    content: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None


class StylometryInfo(BaseModel):
    sentence_length_variance: float
    repetition_score: float
    lexical_diversity: float


class FeaturesInfo(BaseModel):
    perplexity: float
    stylometry: StylometryInfo


class SignalInfo(BaseModel):
    source: str
    verified: bool
    confidence: float


class MetadataInfo(BaseModel):
    model: str
    latency_ms: int
    timestamp: str
    raw_metadata: Optional[dict] = None


class AnalysisResponse(BaseModel):
    truth_score: float
    ai_generated_score: float
    bias_score: float
    credibility_score: float
    confidence_score: float
    explanation: str
    category: Optional[str] = None
    source: Optional[str] = "Unknown"
    features: FeaturesInfo
    signals: List[SignalInfo]
    metadata: MetadataInfo
    # New optional fields from extended forensics layers
    audio_score: Optional[float] = None
    news_consistency_score: Optional[float] = None
    ocr_text: Optional[str] = None

class JobResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    result: Optional[AnalysisResponse] = None # For compatibility if already finished
    progress: Optional[str] = None

class AnalysisUnionResponse(BaseModel):
    # This helps the frontend handle both sync (cached) and async (queued) responses
    status: str
    job_id: Optional[str] = None
    result: Optional[AnalysisResponse] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def resolve_content(request: AnalysisRequest, fallback_label: str = "text") -> str:
    """Extract the actual content string from the flexible request body."""
    value = request.content or request.text or request.url
    if not value or not value.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Request body must include 'content', 'text', or 'url'.",
        )
    return value.strip()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "TruthGuard AI Forensics API",
        "docs": "/docs",
        "status": "active"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "truthguard-api"}


@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    content = resolve_content(request)
    logger.info(f"POST /analyze-text — {len(content)} chars")

    try:
        # Cache check
        content_hash = get_text_hash(content)
        cached = get_cached_result("text", content_hash)
        if cached:
            return cached

        engine = get_engine()
        result = engine.analyze(content)
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Analysis returned no result.",
            )
        
        # Cache set
        set_cached_result("text", content_hash, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during text analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-url", response_model=AnalysisResponse)
async def analyze_url(request: AnalysisRequest):
    content = resolve_content(request)
    logger.info(f"POST /analyze-url — input: {content[:80]}...")

    try:
        engine = get_engine()
        result = engine.analyze(content)
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Could not extract content from URL. Please paste text manually.",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during URL analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    logger.info(f"POST /analyze-image — {file.filename}")

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        engine = get_image_engine()
        result = engine.analyze(contents, filename=file.filename)
        return result
    except Exception as e:
        logger.exception("Error during image analysis")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/analyze-video") # Returns Union
async def analyze_video(file: UploadFile = File(...)):
    logger.info(f"POST /analyze-video — {file.filename} ({file.content_type})")

    try:
        allowed = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/avi"}
        if file.content_type not in allowed and not (file.filename or "").lower().endswith((".mp4", ".mov", ".avi", ".webm")):
            raise HTTPException(status_code=400, detail="File must be a video (mp4, mov, avi, webm)")

        contents = await file.read()
        
        # 1. Hashing & Caching
        content_hash = get_content_hash(contents)
        cached = get_cached_result("video", content_hash)
        if cached:
            return {"status": "complete", "result": cached}

        # 2. Async Queueing
        task = process_video_task.delay(contents, file.filename or "upload.mp4", content_hash)
        logger.info(f"Video analysis queued — task_id: {task.id}")

        return {"status": "processing", "job_id": task.id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during video analysis submission")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Query Celery task status."""
    task_result = AsyncResult(job_id, app=celery_app)
    
    if task_result.state == "SUCCESS":
        return {"status": "complete", "result": task_result.result}
    elif task_result.state == "FAILURE":
        return {"status": "failed", "error": str(task_result.info)}
    
    # Check for custom progress meta
    progress = "Processing..."
    if isinstance(task_result.info, dict):
        progress = task_result.info.get("step", "Processing...")

    return {"status": "processing", "progress": progress}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting TruthGuard API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
