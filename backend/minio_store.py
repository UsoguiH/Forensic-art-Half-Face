"""Optional MinIO (S3-compatible) object store for FULL-RESOLUTION gallery originals.

Design: thumbnails live inside Qdrant (instant display, self-contained), while the
full-size originals live here and are streamed on demand. When the MinIO env vars are unset, everything
still works — galleries just keep thumbnails only.

Enable with:
    FACELAB_MINIO_ENDPOINT   e.g. minio:9000  (host:port, no scheme)
    FACELAB_MINIO_ACCESS_KEY
    FACELAB_MINIO_SECRET_KEY
    FACELAB_MINIO_BUCKET     (default: facelab)
    FACELAB_MINIO_SECURE     1 for https (default 0)
"""
from __future__ import annotations

import io
import os
import threading
from typing import Any

from PIL import Image

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {}

def _cfg() -> dict[str, Any]:
    return {
        "endpoint": os.environ.get("FACELAB_MINIO_ENDPOINT", "").strip(),
        "access": os.environ.get("FACELAB_MINIO_ACCESS_KEY", "").strip(),
        "secret": os.environ.get("FACELAB_MINIO_SECRET_KEY", "").strip(),
        "bucket": (os.environ.get("FACELAB_MINIO_BUCKET", "facelab").strip() or "facelab"),
        "secure": os.environ.get("FACELAB_MINIO_SECURE", "0").lower() in ("1", "true", "yes", "on"),
    }

def _client():
    with _LOCK:
        if "client" in _STATE:
            return _STATE["client"]
        c = _cfg()
        if not (c["endpoint"] and c["access"] and c["secret"]):
            _STATE["client"] = None
            _STATE["error"] = "MinIO env not configured."
            return None
        try:
            from minio import Minio

            client = Minio(c["endpoint"], access_key=c["access"], secret_key=c["secret"], secure=c["secure"])
            if not client.bucket_exists(c["bucket"]):
                client.make_bucket(c["bucket"])
            _STATE["client"] = client
            _STATE["bucket"] = c["bucket"]
            _STATE["error"] = None
            return client
        except Exception as exc:
            _STATE["client"] = None
            _STATE["error"] = f"{type(exc).__name__}: {exc}"
            return None

def enabled() -> bool:
    return _client() is not None

def status() -> dict[str, Any]:
    ok = enabled()
    c = _cfg()
    return {"enabled": ok, "endpoint": c["endpoint"] or None, "bucket": c["bucket"],
            "error": None if ok else _STATE.get("error")}

def put_image(slug: str, key: str, img: Image.Image) -> str | None:
    """Store one full-resolution original. Returns its object name, or None if disabled."""
    client = _client()
    if client is None:
        return None
    obj = f"galleries/{slug}/{key}.jpg"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    size = buf.tell()
    buf.seek(0)
    with _LOCK:
        client.put_object(_STATE["bucket"], obj, buf, size, content_type="image/jpeg")
    return obj

def get_object(obj: str) -> tuple[bytes, str]:
    """Fetch a stored original by object name."""
    client = _client()
    if client is None:
        raise RuntimeError("MinIO is not configured.")
    with _LOCK:
        resp = client.get_object(_STATE["bucket"], obj)
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
    return data, "image/jpeg"
