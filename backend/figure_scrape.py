
from __future__ import annotations

import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def _sim_threshold() -> float:
    """Cosine threshold for 'same person' on buffalo_l ArcFace. Env-tunable."""
    try:
        return float(os.environ.get("FACELAB_FIGURE_SIM", "0.42"))
    except Exception:
        return 0.42

def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿݐ-ݿ]", text or ""))

def _search_image_urls(query: str, max_results: int = 40) -> list[str]:
    DDGS = None
    try:
        from ddgs import DDGS as _D
        DDGS = _D
    except Exception:
        try:
            from duckduckgo_search import DDGS as _D
            DDGS = _D
        except Exception:
            DDGS = None
    if DDGS is None:
        return []

    urls: list[str] = []
    try:
        with DDGS() as ddgs:
            for row in ddgs.images(query, safesearch="moderate", max_results=max_results):
                url = row.get("image") or row.get("thumbnail") or row.get("url")
                if url and url.startswith("http"):
                    urls.append(url)
    except Exception:
        try:
            with DDGS() as ddgs:
                for row in ddgs.images(keywords=query, max_results=max_results):
                    url = row.get("image") or row.get("thumbnail")
                    if url and url.startswith("http"):
                        urls.append(url)
        except Exception:
            return urls
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]

def _download_image(url: str, timeout: float = 6.0, max_bytes: int = 12_000_000) -> Image.Image | None:
    import requests

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _UA}, stream=True)
        resp.raise_for_status()
        content = resp.content
        if not content or len(content) > max_bytes:
            return None
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")
    except Exception:
        return None

def _frontal_score(face: Any) -> float:
    """0..1, higher = more frontal, from the 5-point landmarks."""
    try:
        kps = np.asarray(face.kps, dtype=np.float32)
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_mid = (left_eye + right_eye) / 2.0
        inter_eye = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
        offset = abs(float(nose[0] - eye_mid[0])) / inter_eye
        return float(max(0.0, 1.0 - min(1.0, offset / 0.6)))
    except Exception:
        return 0.5

def _crop_face(image: Image.Image, face: Any, margin: float = 0.45, out_size: int = 768) -> Image.Image:
    """Crop a square region around the detected face with generous margin."""
    W, H = image.size
    x0, y0, x1, y1 = [float(v) for v in face.bbox]
    fw, fh = (x1 - x0), (y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(fw, fh) * (0.5 + margin)
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(W, cx + half))
    bottom = int(min(H, cy + half))
    crop = image.crop((left, top, right, bottom))
    side = min(out_size, max(crop.size))
    return crop.resize((side, side), Image.Resampling.LANCZOS)

def _dominant_identity(embeddings: np.ndarray, sim_threshold: float) -> tuple[np.ndarray, int]:
    """Return (keep_mask, seed_support) for the largest mutually-similar cluster.

    `embeddings` is (M, 512), unit-norm, so `embeddings @ embeddings.T` is the
    cosine-similarity matrix.  The searched person is the face that agrees with
    the most other faces; we grow a consensus centroid from its neighbourhood and
    keep every face within `sim_threshold` of it.
    """
    M = embeddings.shape[0]
    if M <= 1:
        return np.ones(M, dtype=bool), M

    sims = embeddings @ embeddings.T
    support = (sims >= sim_threshold).sum(axis=1)
    seed = int(support.argmax())

    keep = sims[seed] >= sim_threshold
    centroid = embeddings[keep].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    for _ in range(4):
        new_keep = (embeddings @ centroid) >= sim_threshold
        if new_keep.sum() == 0:
            break
        centroid = embeddings[new_keep].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if not keep.any():
        keep = np.zeros(M, dtype=bool)
        keep[seed] = True
    return keep, int(support.max())

def _make_candidate(image: Image.Image, faces: list, url: str, min_face_px: int) -> dict[str, Any] | None:
    """Build one scored candidate from a detected image, or None if unusable."""
    if not faces:
        return None
    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    fw = float(face.bbox[2] - face.bbox[0])
    fh = float(face.bbox[3] - face.bbox[1])
    if min(fw, fh) < min_face_px:
        return None
    vec = np.asarray(getattr(face, "normed_embedding", None), dtype=np.float32)
    if vec is None or vec.size != 512:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    vec = vec / norm
    det = float(getattr(face, "det_score", 0.5) or 0.5)
    frontal = _frontal_score(face)
    solo_bonus = 1.15 if len(faces) == 1 else 1.0
    score = det * ((fw * fh) ** 0.25) * (0.5 + 0.5 * frontal) * solo_bonus
    return {
        "image": _crop_face(image, face),
        "signature": vec,
        "source": url,
        "score": float(score),
        "frontal": round(frontal, 3),
    }

def fetch_figure_faces(engine: Any, name: str, want: int = 6, per_query: int = 24,
                       min_face_px: int = 70, max_faces: int = 18, workers: int = 8) -> dict[str, Any]:
    """Return real reference crops of the SAME public figure `name`.

    Speed: image downloads (the slow, network-bound part) run in a thread pool,
    queries are issued lazily, and the search stops as soon as the dominant
    identity already has `want` photos -- so a well-known name usually finishes on
    the first query without fetching everything.

    Result dict:
      candidates    : up to `want` items, each
                      {"image": PIL.Image, "signature": np.ndarray(512),
                       "source": url, "score": float, "frontal": float},
                      all of the dominant identity, best-quality first.
      found_faces   : how many usable faces were detected.
      identity_kept : how many belonged to the dominant identity.
      confident     : whether that identity recurred (>= 2 agreeing photos).
    Requires a working face detector (real mode).
    """
    empty = {"candidates": [], "found_faces": 0, "identity_kept": 0, "confident": False}
    name = (name or "").strip()
    if not name:
        return empty
    try:
        workers = max(1, int(os.environ.get("FACELAB_FIGURE_WORKERS", workers)))
    except Exception:
        pass

    if _has_arabic(name):
        queries = [name, f"{name} صورة", f"{name} وجه", f"{name} بورتريه"]
    else:
        queries = [name, f"{name} portrait", f"{name} face", f"{name} headshot"]

    candidates: list[dict[str, Any]] = []
    tried: set[str] = set()
    threshold = _sim_threshold()

    def dominant_count() -> int:
        if len(candidates) < 2:
            return len(candidates)
        embeddings = np.stack([c["signature"] for c in candidates]).astype(np.float32)
        keep_mask, _ = _dominant_identity(embeddings, threshold)
        return int(keep_mask.sum())

    for query in queries:
        if len(candidates) >= max_faces:
            break
        urls = [u for u in _search_image_urls(query, max_results=per_query) if u not in tried]
        for u in urls:
            tried.add(u)
        urls = urls[: max_faces * 2]
        if not urls:
            continue

        with ThreadPoolExecutor(max_workers=workers) as pool:
            downloaded = list(pool.map(lambda u: (u, _download_image(u)), urls))
        for url, image in downloaded:
            if image is None:
                continue
            candidate = _make_candidate(image, engine.detect_faces(image), url, min_face_px)
            if candidate:
                candidates.append(candidate)
                if len(candidates) >= max_faces:
                    break

        if len(candidates) >= max(10, want + 4) and dominant_count() >= want:
            break

    found_faces = len(candidates)
    if not candidates:
        return empty

    embeddings = np.stack([c["signature"] for c in candidates]).astype(np.float32)
    keep_mask, seed_support = _dominant_identity(embeddings, threshold)
    kept = [c for c, k in zip(candidates, keep_mask) if k]
    kept.sort(key=lambda c: c["score"], reverse=True)

    return {
        "candidates": kept[:want],
        "found_faces": found_faces,
        "identity_kept": len(kept),
        "confident": bool(seed_support >= 2),
    }
