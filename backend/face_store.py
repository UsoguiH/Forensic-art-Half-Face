"""Qdrant-backed store for ArcFace identity vectors."""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

COLLECTION = os.environ.get("FACELAB_QDRANT_COLLECTION", "facelab_faces")
DIM = 512

class FaceStore:
    """Lazy Qdrant client wrapper (embedded local or remote)."""

    def __init__(self, default_path: str | os.PathLike[str] | None = None) -> None:
        self._client: Any | None = None
        self._mode: str | None = None
        self._lock = threading.RLock()
        self._default_path = str(default_path) if default_path else "./qdrant_faces"

    def _connect(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError as exc:
                raise RuntimeError(
                    "qdrant-client is not installed. Run: pip install qdrant-client"
                ) from exc

            url = os.environ.get("FACELAB_QDRANT_URL", "").strip()
            if url:
                api_key = os.environ.get("FACELAB_QDRANT_API_KEY", "").strip() or None
                client = QdrantClient(url=url, api_key=api_key, timeout=30)
                self._mode = "remote"
            else:
                path = os.environ.get("FACELAB_QDRANT_PATH", "").strip() or self._default_path
                Path(path).mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=path)
                self._mode = "embedded"

            if not client.collection_exists(COLLECTION):
                client.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
                )
            self._client = client
            return client

    @staticmethod
    def _as_vector(vector: Any) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size != DIM:
            raise ValueError(f"A face signature must have {DIM} dimensions, got {vec.size}.")
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError("The face signature has zero norm.")
        return vec / norm

    @staticmethod
    def _item(pid: Any, payload: dict | None, vector: Any, score: float | None) -> dict[str, Any]:
        payload = payload or {}
        signature = None
        if vector is not None:
            signature = [round(float(x), 7) for x in np.asarray(vector, dtype=np.float32).reshape(-1)]
        return {
            "id": str(pid),
            "name": str(payload.get("name") or ""),
            "image": payload.get("image"),
            "created_at": float(payload.get("created_at") or 0.0),
            "signature": signature,
            "score": None if score is None else round(float(score), 7),
        }

    def status(self) -> dict[str, Any]:
        try:
            self._connect()
            return {
                "ready": True,
                "backend": "qdrant",
                "mode": self._mode,
                "collection": COLLECTION,
                "count": self.count(),
            }
        except Exception as exc:
            return {
                "ready": False,
                "backend": "qdrant",
                "mode": self._mode,
                "collection": COLLECTION,
                "count": 0,
                "error": str(exc),
            }

    def count(self) -> int:
        client = self._connect()
        with self._lock:
            return int(client.count(COLLECTION, exact=True).count)

    def store(self, vector: Any, name: str, image_data_url: str | None) -> dict[str, Any]:
        from qdrant_client.models import PointStruct

        client = self._connect()
        vec = self._as_vector(vector)
        point_id = str(uuid.uuid4())
        payload = {
            "name": (name or "").strip(),
            "image": image_data_url,
            "created_at": time.time(),
        }
        with self._lock:
            client.upsert(
                collection_name=COLLECTION,
                points=[PointStruct(id=point_id, vector=vec.tolist(), payload=payload)],
            )
        return {"id": point_id, "name": payload["name"], "created_at": payload["created_at"]}

    def similar(self, vector: Any, limit: int = 12) -> list[dict[str, Any]]:
        client = self._connect()
        vec = self._as_vector(vector)
        with self._lock:
            hits = client.query_points(
                collection_name=COLLECTION,
                query=vec.tolist(),
                limit=max(1, int(limit)),
                with_payload=True,
                with_vectors=True,
            ).points
        return [self._item(h.id, h.payload, h.vector, h.score) for h in hits]

    def list(self, limit: int = 12) -> list[dict[str, Any]]:
        client = self._connect()
        limit = max(1, int(limit))
        with self._lock:
            points, _next = client.scroll(
                collection_name=COLLECTION,
                limit=max(limit, 256),
                with_payload=True,
                with_vectors=True,
            )
        items = [self._item(p.id, p.payload, p.vector, None) for p in points]
        items.sort(key=lambda item: item["created_at"], reverse=True)
        return items[:limit]

    def delete(self, point_id: str) -> None:
        from qdrant_client.models import PointIdsList

        client = self._connect()
        with self._lock:
            client.delete(
                collection_name=COLLECTION,
                points_selector=PointIdsList(points=[str(point_id)]),
            )
