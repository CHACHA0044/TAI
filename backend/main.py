import logging
import os
import tempfile
import uuid as _uuid

# Hugging Face cache - prefer environment variable, default to local project cache
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
import sys
import time

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from enum import Enum
import uvicorn

# Celery / Redis are optional — only needed for async video processing.
# Text, URL and image analysis work without them.
try:
    from celery.result import AsyncResult
    from celery_app import celery_app
    from inference.tasks import process_video_task
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    logger_boot = logging.getLogger("truthguard")
    logger_boot.warning(
        "celery not installed — async video processing disabled. "
        "Install with: pip install celery redis"
    )

from utils.cache import get_cached_result, set_cached_result
from utils.hashing import get_content_hash, get_text_hash
from utils.db import db

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

# ------------------------------------------------------------------
# CORS setup
# ------------------------------------------------------------------
frontend_url = os.getenv("FRONTEND", "*")
origins = [frontend_url]
if frontend_url != "*":
    # Add localhost for development convenience
    origins.extend(["http://localhost:3000", "http://localhost:8000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Credentials (cookies/auth headers) must NOT be sent with wildcard origins
    allow_credentials=(frontend_url != "*"),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    from services.warmup_service import start_warmup_service
    start_warmup_service()
    logger.info("Warmup service launched on startup")
    await db.connect()

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
    burstiness: Optional[float] = 0.0


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


class PrimaryVerdict(str, Enum):
    VERIFIED_FACT = "VERIFIED_FACT"
    FALSE_FACT = "FALSE_FACT"
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"
    OPINION = "OPINION"
    BIASED_CONTENT = "BIASED_CONTENT"
    MANIPULATIVE_CONTENT = "MANIPULATIVE_CONTENT"
    SATIRE_OR_SARCASM = "SATIRE_OR_SARCASM"
    CONSPIRACY_OR_EXTRAORDINARY_CLAIM = "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"
    LIKELY_AI_GENERATED = "LIKELY_AI_GENERATED"
    MIXED_ANALYSIS = "MIXED_ANALYSIS"
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"
    EDITED = "EDITED"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"


class DimensionsInfo(BaseModel):
    truth_score: int = 0
    verifiability: int = 0
    ai_likelihood: int = 0
    bias_score: int = 0
    manipulation_score: int = 0
    sarcasm_score: int = 0
    opinion_score: int = 0
    sarcasm: bool = False
    conspiracy_flag: bool = False


class TruthExpandedInfo(BaseModel):
    explanation: str = ""
    evidence: str = ""
    sources: List[str] = Field(default_factory=list)


class AiExpandedInfo(BaseModel):
    explanation: str = ""
    indicators: List[str] = Field(default_factory=list)


class BiasExpandedInfo(BaseModel):
    explanation: str = ""
    indicators: List[str] = Field(default_factory=list)


class ExpandedAnalysisInfo(BaseModel):
    truth_score: TruthExpandedInfo = Field(default_factory=TruthExpandedInfo)
    ai_likelihood: AiExpandedInfo = Field(default_factory=AiExpandedInfo)
    bias_score: BiasExpandedInfo = Field(default_factory=BiasExpandedInfo)
    manipulation_score: BiasExpandedInfo = Field(default_factory=BiasExpandedInfo)
    opinion_score: BiasExpandedInfo = Field(default_factory=BiasExpandedInfo)
    verifiability: BiasExpandedInfo = Field(default_factory=BiasExpandedInfo)


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
    scene_description: Optional[str] = None
    detected_objects: Optional[List[str]] = None
    style: Optional[str] = None
    authenticity_signals: Optional[dict] = None
    why: Optional[str] = None
    if_uncertain: Optional[str] = None
    # Rich analysis fields
    primary_verdict: PrimaryVerdict = PrimaryVerdict.MIXED_ANALYSIS
    triggered_rule: Optional[str] = None
    claim_type: Optional[str] = None
    confidence: int = 0
    dimensions: DimensionsInfo = Field(default_factory=DimensionsInfo)
    dimension_buckets: Optional[dict] = None
    expanded_analysis: ExpandedAnalysisInfo = Field(default_factory=ExpandedAnalysisInfo)
    debug: Optional[dict] = None
    verdict: str = "Inconclusive"
    risk_level: str = "Medium"
    recommendation: str = "Verify against trusted sources."
    key_factors: List[str] = Field(default_factory=list)

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
        # Persistence
        await db.save_result("text", content_hash, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during text analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-url", response_model=AnalysisResponse)
async def analyze_url(request: AnalysisRequest):
    original_url = resolve_content(request)
    logger.info(f"POST /analyze-url — input: {original_url[:80]}...")

    try:
        engine = get_engine()
        result = engine.analyze(original_url)
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Could not extract content from URL. Please paste text manually.",
            )
        # Surface the original URL so the frontend can display it
        if original_url.startswith("http://") or original_url.startswith("https://"):
            result["source"] = original_url
        
        # Persistence
        content_hash = get_text_hash(original_url) # Using URL as key for this entry
        await db.save_result("url", content_hash, result)
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
        
        # Persistence
        content_hash = get_content_hash(contents)
        await db.save_result("image", content_hash, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during image analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-video")  # Returns Union
async def analyze_video(file: UploadFile = File(...)):
    logger.info(f"POST /analyze-video — {file.filename} ({file.content_type})")

    if not HAS_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Video analysis requires celery+redis. Install with: pip install celery redis"
        )

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

        # 2. Save upload to a temp file and queue only its path — never broker raw bytes.
        #    This avoids JSON-serialisation failures and huge Redis memory usage.
        ext = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
        tmp_path = os.path.join(tempfile.gettempdir(), f"tg_upload_{_uuid.uuid4().hex}{ext}")
        with open(tmp_path, "wb") as fh:
            fh.write(contents)
        logger.info(f"Upload saved to temp path: {tmp_path}")

        # 3. Async Queueing — pass path + hash only, not bytes
        task = process_video_task.delay(tmp_path, file.filename or "upload.mp4", content_hash)
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
    if not HAS_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Job status requires celery+redis. Install with: pip install celery redis"
        )

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
