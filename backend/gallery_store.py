"""Qdrant-backed image galleries.

Upload ANY dataset of face images and search it with the consensus search. Each
face's ArcFace vector AND a small thumbnail are stored together as a Qdrant point,
so a gallery is fully self-contained inside Qdrant — no separate object store and
no file-path resolution (which is what made prebuilt indexes fail to display).

One embedded Qdrant instance lives under <INDEX_DIR>/qdrant_galleries (separate
from the face vault). Set FACELAB_QDRANT_URL to use a remote/Docker Qdrant instead.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

DIM = 512
PREFIX = "gallery_"
_LOCK = threading.RLock()
_STATE: dict[str, Any] = {}

def _client():
    with _LOCK:
        if "client" in _STATE:
            return _STATE["client"]
        from qdrant_client import QdrantClient

        url = os.environ.get("FACELAB_QDRANT_URL", "").strip()
        if url:
            client = QdrantClient(url=url, api_key=os.environ.get("FACELAB_QDRANT_API_KEY") or None, timeout=60)
        else:
            base = os.environ.get("FACELAB_GALLERY_QDRANT", "").strip()
            if not base:
                idx = Path(os.environ.get("FACELAB_INDEX_DIR", "../data")).expanduser()
                base = str(idx / "qdrant_galleries")
            Path(base).mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=base)
        _STATE["client"] = client
        return client

def _coll(slug: str) -> str:
    return PREFIX + slug

def exists(slug: str) -> bool:
    try:
        return bool(_client().collection_exists(_coll(slug)))
    except Exception:
        return False

def count(slug: str) -> int:
    try:
        return int(_client().count(_coll(slug), exact=True).count)
    except Exception:
        return 0

def collections() -> list[dict[str, Any]]:
    try:
        cols = _client().get_collections().collections
    except Exception:
        return []
    out = []
    for c in cols:
        if c.name.startswith(PREFIX):
            slug = c.name[len(PREFIX):]
            out.append({"name": slug, "count": count(slug)})
    return out

def _thumb_data_url(img: Image.Image, size: int = 200) -> str:
    t = img.copy()
    t.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    t.convert("RGB").save(buf, format="WEBP", quality=82)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def build(slug: str, files: list, embed: Callable[[Image.Image], "np.ndarray | None"],
          thumb: int = 200, progress: Callable[[int, int], None] | None = None) -> int:
    """(Re)build a Qdrant gallery: embed each image, store vector + thumbnail. Returns kept."""
    from qdrant_client.models import Distance, VectorParams, PointStruct

    try:
        import minio_store
        use_minio = minio_store.enabled()
    except Exception:
        minio_store = None
        use_minio = False

    client = _client()
    coll = _coll(slug)
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    total = len(files)
    kept = 0
    batch: list = []
    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
            v = embed(img)
        except Exception:
            v = None
        if v is not None:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                v = v / nrm
                pid = f"{Path(p).parent.name}/{Path(p).stem}" if Path(p).stem else Path(p).name
                point_id = str(uuid.uuid4())
                payload = {"pid": pid, "image": _thumb_data_url(img, thumb)}
                if use_minio:
                    try:
                        obj = minio_store.put_image(slug, point_id, img)
                        if obj:
                            payload["full"] = obj
                    except Exception:
                        pass
                batch.append(PointStruct(id=point_id, vector=v.tolist(), payload=payload))
                kept += 1
                if len(batch) >= 64:
                    client.upsert(coll, points=batch); batch = []
        if progress and (i % 10 == 0 or i == total - 1):
            progress(i + 1, kept)
    if batch:
        client.upsert(coll, points=batch)
    return kept

def search(slug: str, query_vec, k: int = 18, shortlist: int = 3000):
    """Rank a Qdrant gallery by cosine. Returns (leads, gallery_count, shortlist_count)."""
    client = _client()
    coll = _coll(slug)
    vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    vec = vec / (float(np.linalg.norm(vec)) or 1.0)
    total = count(slug)
    limit = max(k, min(shortlist, max(total, 1)))
    hits = client.query_points(coll, query=vec.tolist(), limit=limit, with_payload=True).points
    best: dict[str, tuple[float, Any, Any]] = {}
    order: list[str] = []
    for h in hits:
        payload = h.payload or {}
        pid = str(payload.get("pid") or h.id)
        score = float(h.score)
        if pid not in best:
            best[pid] = (score, payload.get("image"), payload.get("full"))
            order.append(pid)
        elif score > best[pid][0]:
            best[pid] = (score, best[pid][1], best[pid][2])
    leads = []
    for rank, pid in enumerate(order[:k], 1):
        score, image, full = best[pid]
        leads.append({"rank": rank, "id": pid, "score": round(score, 7),
                      "image": image, "full": full, "you": False})
    return leads, total, len(order)

def delete(slug: str) -> bool:
    try:
        c = _client()
        if c.collection_exists(_coll(slug)):
            c.delete_collection(_coll(slug))
        return True
    except Exception:
        return False
