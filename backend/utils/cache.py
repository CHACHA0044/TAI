import os
import json
import logging

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger("truthguard.cache")

# Global Redis client
_redis_client = None

def get_redis():
    if not HAS_REDIS:
        return None
    global _redis_client
    if _redis_client is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.StrictRedis.from_url(redis_url, decode_responses=True)
    return _redis_client

def get_cached_result(prefix: str, content_hash: str):
    """Fetch cached dict from Redis via {prefix}:{hash} if exists."""
    try:
        r = get_redis()
        key = f"{prefix}:{content_hash}"
        cached = r.get(key)
        if cached:
            logger.info(f"CACHE HIT: {key}")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis cache check failed: {e}")
    return None

def set_cached_result(prefix: str, content_hash: str, payload: dict, expire_s: int = 86400):
    """Store result dict to Redis and set 24h expiration."""
    try:
        r = get_redis()
        key = f"{prefix}:{content_hash}"
        r.set(key, json.dumps(payload), ex=expire_s)
        logger.info(f"CACHE SET: {key}")
    except Exception as e:
        logger.warning(f"Redis cache set failed: {e}")
