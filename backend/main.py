import logging
import os

# Set Hugging Face cache to D: drive to avoid C: drive usage
os.environ["HF_HOME"] = "d:/rep/tai/.cache/huggingface"
import sys
import time

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

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
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "truthguard-api"}


@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    content = resolve_content(request)
    logger.info(f"POST /analyze-text — {len(content)} chars")

    try:
        engine = get_engine()
        result = engine.analyze(content)
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Analysis returned no result. Content may be too short or extraction failed.",
            )
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting TruthGuard API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
