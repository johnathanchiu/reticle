"""Image loading and media type utilities."""

import base64
from pathlib import Path


def load_image_base64(path: str) -> str:
    """Load an image file and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def infer_media_type(path: str) -> str:
    """Infer MIME type from file extension."""
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
