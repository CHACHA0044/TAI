import hashlib

def get_content_hash(content: bytes) -> str:
    """Generate SHA256 hash for bytes."""
    return hashlib.sha256(content).hexdigest()

def get_text_hash(text: str) -> str:
    """Generate SHA256 hash for string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
