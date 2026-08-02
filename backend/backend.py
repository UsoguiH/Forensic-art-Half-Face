"""Face Lab FastAPI backend and static web server."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import identity_studio as studio
import enroll as enroll_mod
from face_store import FaceStore

ENGINE = studio.ENGINE
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = Path(os.environ.get("FACELAB_STATIC_DIR", BASE_DIR.parent / "static")).resolve()
INDEX_DIR = Path(os.environ.get("FACELAB_INDEX_DIR", BASE_DIR.parent / "data")).resolve()
FACES = FaceStore(default_path=INDEX_DIR / "qdrant_faces")
MAX_DATA_URL_CHARS = int(os.environ.get("FACELAB_MAX_DATA_URL", 24_000_000))
MODEL_LOCK = threading.RLock()

app = FastAPI(title="Face Lab API", version="2.0.0")
allowed_origins = [x.strip() for x in os.environ.get("FACELAB_CORS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _warmup() -> None:
    """Load the diffusion pipeline + ArcFace up front, in the background, so the
    first user-facing action is instant instead of paying the cold-load cost.
    Disable with FACELAB_WARMUP=0.  No-op in mock mode."""
    if os.environ.get("FACELAB_WARMUP", "1") == "0" or getattr(ENGINE, "is_mock", False):
        return

    def run() -> None:
        try:
            with MODEL_LOCK:
                if getattr(ENGINE, "provider", "") in ("fal", "openrouter"):
                    # A hosted API has no cold-load cost to amortise, so generating
                    # here would only spend credits on every restart. ArcFace does
                    # load locally, so warm just that.
                    ENGINE.signature(Image.new("RGB", (448, 448), (127, 127, 127)))
                    return
                img = ENGINE.generate("a passport-style studio portrait", seed=1, width=448, height=448)
                ENGINE.signature(img)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()

def to_data_url(img: Image.Image, *, quality: int = 92, lossless: bool = False) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=quality, lossless=lossless, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def from_data_url(value: str) -> Image.Image:
    if not value or len(value) > MAX_DATA_URL_CHARS:
        raise ValueError("Image is empty or exceeds the configured upload limit.")
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid base64 image.") from exc

def vec_list(vec: np.ndarray | None) -> list[float] | None:
    if vec is None:
        return None
    return [round(float(x), 7) for x in np.asarray(vec, dtype=np.float32).reshape(-1)]

_FEMALE = {"female", "woman", "f", "أنثى", "امرأة", "انثى", "نساء"}
_MALE = {"male", "man", "m", "ذكر", "رجل"}

def _gender_phrase(gender: str | None) -> str:
    """Turn a gender hint into a prompt prefix so generation isn't male-by-default."""
    g = (gender or "").strip().lower()
    if g in _FEMALE:
        return "a woman, "
    if g in _MALE:
        return "a man, "
    return ""

def require_signature(image: Image.Image, label: str = "image") -> np.ndarray:
    vector = ENGINE.signature(image)
    if vector is None:
        raise ValueError(f"No detectable face was found in the {label}.")
    return vector

def fail(exc: Exception, status: int = 500) -> HTTPException:
    return HTTPException(status_code=status, detail=f"{type(exc).__name__}: {exc}")

class GenReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)
    steps: int | None = Field(default=None, ge=1, le=80)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)
    public_figure: str | None = Field(default=None, max_length=200)
    gender: str | None = Field(default=None, max_length=16)
    model: str | None = Field(default=None, pattern="^(dev|klein)$")

class EditReq(BaseModel):
    image: str
    instruction: str = Field(default="", max_length=3000)
    ref: str
    mode: str = Field(default="preserve", pattern="^preserve$")
    steps: int | None = Field(default=None, ge=1, le=80)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    anchor_signature: list[float] | None = None
    mask: dict[str, Any] | None = None
    model: str | None = Field(default=None, pattern="^(dev|klein)$")

class FigureReq(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    count: int = Field(default=6, ge=1, le=12)

class SigReq(BaseModel):
    image: str

class LandmarksReq(BaseModel):
    image: str

class SimReq(BaseModel):
    a: list[float]
    b: list[float]

class FaceStoreReq(BaseModel):
    image: str
    signature: list[float] | None = None
    name: str = Field(default="", max_length=200)

class FaceQueryReq(BaseModel):
    signature: list[float] | None = None
    limit: int = Field(default=12, ge=1, le=50)

class FaceDeleteReq(BaseModel):
    id: str = Field(min_length=1, max_length=64)

class EnrollReq(BaseModel):
    images: list[str] = Field(min_length=1, max_length=6)
    side_images: list[str] = Field(default_factory=list, max_length=6)
    label: str = Field(default="ME", max_length=60)

class SearchReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)
    n: int = Field(default=30, ge=2, le=40)
    k: int = Field(default=18, ge=1, le=50)
    side_weight: float = Field(default=0.7, ge=0.0, le=2.0)
    gender: str | None = Field(default=None, max_length=16)
    fast: bool | None = None

class SearchImageReq(BaseModel):
    image: str
    signature: list[float] | None = None
    k: int = Field(default=18, ge=1, le=50)

class TranscribeReq(BaseModel):
    audio: str
    language: str = Field(default="ar", pattern="^(ar|en)$")

SEARCH_FILES = {
    "front_index": "front.index",
    "front_ids": "front_ids.npy",
    "side_index": "side.index",
    "side_ids": "side_ids.npy",
    "front_paths": "front_paths.json",
}

SEARCH_REQUIRED = ("front_index", "front_ids", "front_paths")

_GALLERY_ROOT = INDEX_DIR / "galleries"
_ACTIVE_DIR = INDEX_DIR
_ACTIVE_QDRANT: str | None = None

def _active_dir() -> Path:
    return _ACTIVE_DIR

def _active_is_qdrant() -> bool:
    return _ACTIVE_QDRANT is not None

def search_paths() -> dict[str, Path]:
    base = _active_dir()
    return {name: base / filename for name, filename in SEARCH_FILES.items()}

def _module_present(stem: str) -> bool:
    return (BASE_DIR / f"{stem}.py").exists() or (BASE_DIR / f"{stem}.pyc").exists()

def search_status() -> dict[str, Any]:
    modules = [f"{stem}.py" for stem in ("two_view_search", "face_pipeline")
               if not _module_present(stem)]
    if _active_is_qdrant():
        try:
            import gallery_store as gs
            ok = gs.exists(_ACTIVE_QDRANT) and gs.count(_ACTIVE_QDRANT) > 0
        except Exception:
            ok = False
        return {
            "ready": ok and not modules and not ENGINE.is_mock,
            "index_dir": f"qdrant:{_ACTIVE_QDRANT}",
            "missing": ([] if ok else ["qdrant gallery"]) + modules,
        }
    paths = search_paths()
    missing = [paths[k].name for k in SEARCH_REQUIRED if not paths[k].exists()]
    return {
        "ready": not missing and not modules and not ENGINE.is_mock,
        "index_dir": str(_active_dir()),
        "missing": missing + modules,
    }

@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail=f"Frontend not found at {index}")
    return FileResponse(index)

@app.get("/health")
def health():
    status = ENGINE.status()
    status.update(
        ok=True,
        search=search_status(),
        faces=FACES.status(),
        enroll=enroll_mod.status(INDEX_DIR),
        api_version="2.3.0",
    )
    try:
        import minio_store
        status["minio"] = minio_store.status()
    except Exception:
        status["minio"] = {"enabled": False}
    return status

@app.post("/enroll")
def enroll_face(req: EnrollReq):
    """Plant a real face into the 70k gallery so a genuine text search can rank it."""
    try:
        front = [from_data_url(x) for x in req.images]
        side = [from_data_url(x) for x in req.side_images] if req.side_images else None
        with MODEL_LOCK:
            result = enroll_mod.enroll(INDEX_DIR, ENGINE.signature, front, side, req.label)
        _path_map.cache_clear()
        return {**result, "ok": True}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except RuntimeError as exc:
        raise fail(exc, 400) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/enroll/clear")
def enroll_clear():
    try:
        with MODEL_LOCK:
            result = enroll_mod.clear(INDEX_DIR)
        _path_map.cache_clear()
        return {**result, "ok": True}
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/generate")
def generate(req: GenReq):
    prompt = req.prompt.strip()
    if req.public_figure:
        figure = req.public_figure.strip()
        if figure and figure.lower() not in prompt.lower():
            prompt = f"a photorealistic portrait of a person resembling {figure}, {prompt}"
    prompt = _gender_phrase(req.gender) + prompt
    extra: dict[str, Any] = {}
    if req.model and getattr(ENGINE, "remote", False):
        extra["model"] = req.model
    try:
        with MODEL_LOCK:
            image = ENGINE.generate(
                prompt,
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
                width=req.width,
                height=req.height,
                **extra,
            )
            vector = require_signature(image, "generated image")
        return {
            "image": to_data_url(image, lossless=True),
            "signature": vec_list(vector),
            "mock": ENGINE.is_mock,
            "model": ENGINE.status()["image_model"],
        }
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/figure")
def figure(req: FigureReq):
    """Fetch real reference photos of a named public figure to use as an editable base.

    Real mode searches public web images, detects/crops the primary face, and
    returns ranked candidates locked to their real ArcFace signatures.  If no
    usable photo is found (or in mock/offline mode) it falls back to a clearly
    labelled FLUX-generated look-alike.
    """
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Enter a public figure name.")
    try:
        scrape: dict[str, Any] = {}
        if not ENGINE.is_mock:
            try:
                from figure_scrape import fetch_figure_faces

                with MODEL_LOCK:
                    scrape = fetch_figure_faces(ENGINE, name, want=req.count)
            except Exception:
                scrape = {}

        found = scrape.get("candidates") or []
        if found:
            candidates = [
                {
                    "image": to_data_url(item["image"], quality=92),
                    "signature": vec_list(item["signature"]),
                    "source": item["source"],
                    "score": round(float(item["score"]), 4),
                    "frontal": item.get("frontal"),
                }
                for item in found
            ]
            found_faces = int(scrape.get("found_faces", len(candidates)))
            identity_kept = int(scrape.get("identity_kept", len(candidates)))
            dropped = max(0, found_faces - identity_kept)
            confident = bool(scrape.get("confident", True))
            note = f"{len(candidates)} reference photo(s) of {name}"
            if dropped > 0:
                note += f" — filtered out {dropped} other face(s) that appeared in the results"
            note += ". Each is locked to its real ArcFace signature; any edit is synthetic."
            if not confident:
                note = (
                    f"Could not confirm one consistent identity for {name} from web results; "
                    "showing the closest matches — verify before editing. "
                ) + note
            return {
                "name": name,
                "candidates": candidates,
                "count": len(candidates),
                "scraped": True,
                "mock": ENGINE.is_mock,
                "dropped": dropped,
                "confident": confident,
                "note": note,
            }

        prompt = f"a photorealistic portrait of a person resembling {name}, neutral studio lighting"
        with MODEL_LOCK:
            image = ENGINE.generate(prompt, seed=42)
            vector = require_signature(image, "generated look-alike")
        return {
            "name": name,
            "candidates": [
                {
                    "image": to_data_url(image, lossless=True),
                    "signature": vec_list(vector),
                    "source": None,
                    "score": None,
                    "frontal": None,
                }
            ],
            "count": 1,
            "scraped": False,
            "mock": ENGINE.is_mock,
            "note": (
                "No usable public photo was retrieved, so this is a FLUX-generated look-alike "
                f"resembling {name} (a synthetic face, not a real photo)."
            ),
        }
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/signature")
def signature(req: SigReq):
    try:
        image = from_data_url(req.image)
        with MODEL_LOCK:
            vector = require_signature(image)
        return {"signature": vec_list(vector)}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/similarity")
def similarity(req: SimReq):
    score = ENGINE.similarity(req.a, req.b)
    if score is None:
        raise HTTPException(status_code=422, detail="The two signatures are invalid or incompatible.")
    return {"cosine": score}

def _ellipse_poly(cx: float, cy: float, rx: float, ry: float, W: int, H: int, n: int = 18) -> list[list[float]]:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = (cx + rx * math.cos(a)) / W
        y = (cy + ry * math.sin(a)) / H
        pts.append([round(min(1.0, max(0.0, x)), 4), round(min(1.0, max(0.0, y)), 4)])
    return pts

@app.post("/landmarks")
def landmarks(req: LandmarksReq):
    """Return named facial regions (feature-shaped polygons) for smart selection.

    Built from the 5 stable ArcFace keypoints (eyes, nose, mouth corners) scaled
    by interocular distance — robust across detectors.  Empty list when no face
    is detected (mock/offline), in which case the UI keeps freehand selection.
    """
    try:
        image = from_data_url(req.image)
        with MODEL_LOCK:
            faces = ENGINE.detect_faces(image)
        if not faces:
            return {"regions": [], "detected": False}
        W, H = image.size
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        kps = np.asarray(face.kps, dtype=float)
        le, re, nose, lm, rm = kps
        d = float(np.linalg.norm(re - le)) or (W * 0.2)
        mouth_c = (lm + rm) / 2.0
        mw = float(np.linalg.norm(rm - lm))
        regions = [
            {"name": "الحاجب الأيمن", "polygon": _ellipse_poly(le[0], le[1] - 0.44 * d, 0.36 * d, 0.15 * d, W, H)},
            {"name": "الحاجب الأيسر", "polygon": _ellipse_poly(re[0], re[1] - 0.44 * d, 0.36 * d, 0.15 * d, W, H)},
            {"name": "العين اليمنى", "polygon": _ellipse_poly(le[0], le[1], 0.30 * d, 0.18 * d, W, H)},
            {"name": "العين اليسرى", "polygon": _ellipse_poly(re[0], re[1], 0.30 * d, 0.18 * d, W, H)},
            {"name": "الأنف", "polygon": _ellipse_poly(nose[0], nose[1], 0.26 * d, 0.42 * d, W, H)},
            {"name": "الفم", "polygon": _ellipse_poly(mouth_c[0], mouth_c[1], max(0.55 * d, 0.62 * mw), 0.24 * d, W, H)},
            {"name": "الخد الأيمن", "polygon": _ellipse_poly((le[0] + nose[0]) / 2, nose[1] + 0.05 * d, 0.28 * d, 0.36 * d, W, H)},
            {"name": "الخد الأيسر", "polygon": _ellipse_poly((re[0] + nose[0]) / 2, nose[1] + 0.05 * d, 0.28 * d, 0.36 * d, W, H)},
        ]
        return {"regions": regions, "detected": True}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

def _face_thumbnail(image: Image.Image, size: int = 512) -> str:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    return to_data_url(thumb, quality=90)

@app.post("/faces/store")
def faces_store(req: FaceStoreReq):
    """Store a face in the Qdrant vector database.

    The exact 512-d ArcFace signature is saved as the point vector, so a later
    retrieval returns the identical unique facial features, not a re-estimate.
    """
    try:
        image = from_data_url(req.image)
        if req.signature:
            vector = np.asarray(req.signature, dtype=np.float32)
        else:
            with MODEL_LOCK:
                vector = require_signature(image, "face to store")
        entry = FACES.store(vector, req.name, _face_thumbnail(image))
        return {**entry, "count": FACES.count(), "ok": True}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/faces/query")
def faces_query(req: FaceQueryReq):
    """Retrieve stored faces from Qdrant.

    With a signature: nearest faces by cosine similarity, each carrying its
    original stored signature.  Without one: the most recently stored faces.
    """
    try:
        if req.signature:
            faces = FACES.similar(req.signature, req.limit)
            by = "similarity"
        else:
            faces = FACES.list(req.limit)
            by = "recent"
        return {"faces": faces, "by": by, "count": FACES.count(), "ok": True}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/faces/delete")
def faces_delete(req: FaceDeleteReq):
    try:
        FACES.delete(req.id)
        return {"ok": True, "count": FACES.count()}
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/edit")
def edit(req: EditReq):
    extra: dict[str, Any] = {}
    if req.model and getattr(ENGINE, "remote", False):
        extra["model"] = req.model
    try:
        current = from_data_url(req.image)
        reference = from_data_url(req.ref)
        with MODEL_LOCK:
            output = ENGINE.edit(
                current,
                req.instruction,
                reference,
                req.mode,
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
                mask=req.mask,
                **extra,
            )
            output_vector = require_signature(output, "edited image")
            if req.anchor_signature:
                anchor = np.asarray(req.anchor_signature, dtype=np.float32)
            else:
                anchor = require_signature(reference, "reference image")
            score = ENGINE.similarity(output_vector, anchor)
        return {
            "image": to_data_url(output, lossless=True),
            "signature": vec_list(output_vector),
            "similarity": score,
            "mock": ENGINE.is_mock,
            "region_applied": bool(req.mask),
        }
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

def _mean_query(images: list[Image.Image]) -> tuple[np.ndarray, int]:
    vectors = [ENGINE.signature(image) for image in images]
    valid = [v for v in vectors if v is not None]
    if not valid:
        raise RuntimeError("No faces were detected in the generated query images.")
    query = np.mean(np.stack(valid).astype(np.float32), axis=0)
    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise RuntimeError("The averaged query embedding has zero norm.")
    return query / norm, len(valid)

def _variance_dots_from_stack(
    arrays: list[np.ndarray], grid: int = 88, threshold: float = 0.10, step: int = 3, cap: int = 340
) -> list[dict[str, float]]:
    """Per-region luminance variance across a spatially-aligned image stack.

    `arrays` are the SAME uint8 RGB frames that were averaged into the composite
    (canonical landmark-aligned crops in real mode, or the resized raw stack in
    the fallback).  High variance = the region where the generated faces disagreed
    = the region that AVERAGED into a BLUR in the composite.  A dot's intensity
    `v` scales with that local blur, so the overlay is densest/reddest exactly
    where the composite is blurriest and empty where it is sharp.

    Returned in NORMALISED [0,1] composite coordinates.  Fixes: finer grid, a
    lower threshold so faint blur still marks, a smoothed intensity curve, light
    spatial blurring of the variance field (so dots sit on blurred *areas*, not
    speckle), and no arbitrary decimation that previously dropped the reddest.
    """
    if len(arrays) < 2:
        return []
    size = int(grid)
    frames = []
    for arr in arrays:
        img = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert("L").resize((size, size))
        frames.append(np.asarray(img, dtype=np.float32))
    variance = np.stack(frames).var(axis=0)

    vmax = float(variance.max()) or 1e-6
    v8 = Image.fromarray((variance / vmax * 255.0).clip(0, 255).astype(np.uint8), "L")
    variance = np.asarray(v8.filter(ImageFilter.GaussianBlur(radius=1.4)), dtype=np.float32)

    hi = float(np.percentile(variance, 99)) or 1e-6
    norm = np.clip(variance / hi, 0.0, 1.0)
    dots: list[dict[str, float]] = []
    for y in range(1, size - 1, step):
        for x in range(1, size - 1, step):
            value = float(norm[y, x])
            if value > threshold:

                dots.append({"x": (x + 0.5) / size, "y": (y + 0.5) / size, "v": round(value ** 0.7, 4)})

    if len(dots) > cap:
        dots.sort(key=lambda d: d["v"], reverse=True)
        dots = dots[:cap]
    return dots

def _simple_composite_and_dots(
    images: list[Image.Image], size: int = 256
) -> tuple[Image.Image, list[dict[str, float]]]:
    frames = [np.asarray(im.convert("RGB").resize((size, size)), dtype=np.uint8) for im in images]
    composite = Image.fromarray(np.mean(np.stack(frames).astype(np.float32), axis=0).clip(0, 255).astype(np.uint8))
    return composite, _variance_dots_from_stack(frames)

def _composite_and_dots(images: list[Image.Image]) -> tuple[Image.Image, list[dict[str, float]]]:
    """Return (composite, variance_dots) computed from the SAME aligned stack."""
    try:
        from face_pipeline import aligned_average

        arr, _n, crops = aligned_average(images, upsample=256, return_crops=True)
        return Image.fromarray(arr).convert("RGB"), _variance_dots_from_stack(crops)
    except Exception:
        return _simple_composite_and_dots(images)

@lru_cache(maxsize=1)
def _path_map() -> dict[str, str]:
    path = search_paths()["front_paths"]
    with path.open("r", encoding="utf-8") as handle:
        return {str(k): str(v) for k, v in json.load(handle).items()}

_BASENAME_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}

@lru_cache(maxsize=1)
def _basename_map() -> dict[str, str]:
    """Resolve a gallery image by file name when its stored path is absent.

    Lets a prebuilt index (e.g. built on Kaggle, with paths like
    /kaggle/input/.../front/front/K50649) render its faces anywhere, as long as
    FACELAB_DATASET_ROOT points at the images. Mugshot datasets often store files
    with NO extension (e.g. "K50649"), so those are included too.
    """
    root = os.environ.get("FACELAB_DATASET_ROOT", "").strip()
    if not root or not Path(root).exists():
        return {}
    mapping: dict[str, str] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _BASENAME_EXT or suffix == "":
            mapping.setdefault(path.name.lower(), str(path))
    return mapping

def _resolve_image_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return str(path)
    return _basename_map().get(path.name.lower())

def _real_search(prompt: str, n: int, k: int, side_weight: float, fast: bool | None = None) -> dict[str, Any] | None:
    status = search_status()
    if not status["ready"]:
        return None

    import faiss

    if fast is None:
        fast = os.environ.get("FACELAB_FAST", "1") != "0"
    try:
        qsize = int(os.environ.get("FACELAB_QUERY_SIZE", "448" if fast else "640"))
    except Exception:
        qsize = 448 if fast else 640
    qsteps = None
    if os.environ.get("FACELAB_QUERY_STEPS", "").strip():
        try:
            qsteps = int(os.environ["FACELAB_QUERY_STEPS"])
        except Exception:
            qsteps = None
    if fast:

        n = min(n, int(os.environ.get("FACELAB_QUERY_N", "30")))

    front_images = ENGINE.generate_many(
        f"passport-style frontal portrait, looking directly at camera, {prompt}",
        n, steps=qsteps, base_seed=0, width=qsize, height=qsize,
    )
    front_query, front_valid = _mean_query(front_images)
    side_valid = 0

    if _active_is_qdrant():

        import gallery_store as gs
        leads, gallery, results_len = gs.search(_ACTIVE_QDRANT, front_query, k)
        enrolled = set()
        you_rank = you_id = None
        missing_images = 0
    else:
        paths = search_paths()
        front_index = faiss.read_index(str(paths["front_index"]))
        front_ids = np.load(paths["front_ids"], allow_pickle=True)
        gallery = int(front_index.ntotal)
        shortlist_target = min(3000, gallery)
        query = np.ascontiguousarray(np.asarray(front_query, dtype=np.float32).reshape(1, -1))
        sims, idxs = front_index.search(query, shortlist_target)

        best: dict[str, float] = {}
        order: list[str] = []
        for sim, ix in zip(sims[0].tolist(), idxs[0].tolist()):
            if ix < 0:
                continue
            pid = str(front_ids[ix])
            if pid not in best:
                best[pid] = float(sim)
                order.append(pid)
            else:
                best[pid] = max(best[pid], float(sim))
        results = [(pid, best[pid]) for pid in order]
        results_len = len(results)

        enrolled = enroll_mod.enrolled_ids(INDEX_DIR)
        you_rank = None
        you_id = None
        if enrolled:
            for rank, (person_id, _score) in enumerate(results, 1):
                if str(person_id) in enrolled:
                    you_rank, you_id = rank, str(person_id)
                    break

        path_map = _path_map()
        leads = []
        missing_images = 0
        for rank, (person_id, score) in enumerate(results, 1):
            if len(leads) >= k:
                break
            path = _resolve_image_path(path_map.get(str(person_id)))
            if not path:
                missing_images += 1
                continue
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                missing_images += 1
                continue
            leads.append(
                {
                    "rank": rank,
                    "id": str(person_id),
                    "score": round(float(score), 7),
                    "image": to_data_url(image, quality=88),
                    "you": str(person_id) in enrolled,
                }
            )

    composite_img, variance_dots = _composite_and_dots(front_images)
    return {
        "faces": [to_data_url(image, quality=86) for image in front_images],
        "composite": to_data_url(composite_img, quality=90),
        "variance_dots": variance_dots,
        "leads": leads,
        "shortlist": results_len,
        "gallery": gallery,
        "real": True,
        "score_kind": "cosine",
        "you_rank": you_rank,
        "you_id": you_id,
        "you_in_leads": bool(you_rank and you_rank <= k),
        "enrolled": len(enrolled),
        "front_valid": front_valid,
        "side_valid": side_valid,
        "gallery_kind": "qdrant" if _active_is_qdrant() else "faiss",
        "gallery_name": _ACTIVE_QDRANT if _active_is_qdrant() else ("default" if _active_dir() == INDEX_DIR else _active_dir().name),
        "warning": (
            f"{missing_images} ranked gallery images could not be resolved. "
            "Re-run prepare_gallery.py to refresh front_paths.json."
            if missing_images
            else None
        ),
    }

def _mock_search(prompt: str, n: int, k: int) -> dict[str, Any]:
    images = [studio._mock_portrait(prompt, i) for i in range(n)]

    try:
        query, valid = _mean_query(images)
        embed = ENGINE.signature
    except Exception:
        embed = studio._mock_signature
        query = np.mean(np.stack([embed(im) for im in images]).astype(np.float32), axis=0)
        query /= float(np.linalg.norm(query)) or 1.0
        valid = 0
    leads = []
    for i in range(k):
        image = studio._mock_portrait(prompt + "|gallery", 100 + i)
        vector = embed(image)
        score = ENGINE.similarity(query, vector)
        leads.append(
            {
                "rank": i + 1,
                "id": f"DEMO-{1000 + i}",
                "score": round(float(score or 0.0), 7),
                "image": to_data_url(image),
            }
        )
    leads.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(leads, 1):
        item["rank"] = rank
    composite_img, variance_dots = _simple_composite_and_dots(images)
    return {
        "faces": [to_data_url(image) for image in images],
        "composite": to_data_url(composite_img),
        "variance_dots": variance_dots,
        "leads": leads,
        "shortlist": k,
        "gallery": k,
        "real": False,
        "score_kind": "cosine",
        "front_valid": valid,
        "side_valid": 0,
        "warning": "DEMO search: the FAISS gallery is not ready. Restore or build the index for real retrieval.",
    }

@app.post("/search")
def search(req: SearchReq):
    started = time.perf_counter()
    prompt = _gender_phrase(req.gender) + req.prompt
    try:
        with MODEL_LOCK:
            result = _real_search(prompt, req.n, req.k, req.side_weight, req.fast)
            if result is None:
                result = _mock_search(prompt, req.n, req.k)
        result["seconds"] = round(time.perf_counter() - started, 2)
        result["kept_percent"] = round(100.0 * result["shortlist"] / max(1, result["gallery"]), 3)
        result["eliminated_percent"] = round(100.0 - result["kept_percent"], 3)
        return result
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/search_image")
def search_image(req: SearchImageReq):
    """Fully OFFLINE search: embed an uploaded photo with the LOCAL ArcFace model
    and rank the LOCAL gallery index by cosine similarity.  No image generation,
    no internet — works even when the generation server is offline."""
    started = time.perf_counter()
    status = search_status()
    if not status["ready"]:
        raise fail(RuntimeError("The local gallery index is not built yet. Run index_local first."), 400)
    try:
        if req.signature:
            q = np.asarray(req.signature, dtype=np.float32).reshape(-1)
        else:
            image = from_data_url(req.image)
            with MODEL_LOCK:
                q = require_signature(image, "query photo")
        q = q / (float(np.linalg.norm(q)) or 1.0)

        if _active_is_qdrant():
            import gallery_store as gs
            leads, gallery, results_len = gs.search(_ACTIVE_QDRANT, q, req.k)
            return {
                "real": True, "offline": True, "score_kind": "cosine",
                "leads": leads, "gallery": gallery, "shortlist": results_len,
                "you_rank": None, "enrolled": 0,
                "seconds": round(time.perf_counter() - started, 2),
            }

        import faiss

        paths = search_paths()
        front_index = faiss.read_index(str(paths["front_index"]))
        front_ids = np.load(paths["front_ids"], allow_pickle=True)
        gallery = int(front_index.ntotal)
        sims, idxs = front_index.search(
            np.ascontiguousarray(q.reshape(1, -1)), min(3000, gallery)
        )
        best: dict[str, float] = {}
        order: list[str] = []
        for sim, ix in zip(sims[0].tolist(), idxs[0].tolist()):
            if ix < 0:
                continue
            pid = str(front_ids[ix])
            if pid not in best:
                best[pid] = float(sim)
                order.append(pid)
            else:
                best[pid] = max(best[pid], float(sim))
        results = [(pid, best[pid]) for pid in order]

        enrolled = enroll_mod.enrolled_ids(INDEX_DIR)
        you_rank = None
        for rank, (pid, _s) in enumerate(results, 1):
            if str(pid) in enrolled:
                you_rank = rank
                break

        path_map = _path_map()
        leads: list[dict[str, Any]] = []
        missing = 0
        for rank, (pid, score) in enumerate(results, 1):
            if len(leads) >= req.k:
                break
            p = _resolve_image_path(path_map.get(str(pid)))
            if not p:
                missing += 1
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                missing += 1
                continue
            leads.append({
                "rank": rank, "id": str(pid), "score": round(float(score), 7),
                "image": to_data_url(img, quality=88), "you": str(pid) in enrolled,
            })
        return {
            "real": True, "offline": True, "score_kind": "cosine",
            "leads": leads, "gallery": gallery, "shortlist": len(results),
            "you_rank": you_rank, "enrolled": len(enrolled),
            "seconds": round(time.perf_counter() - started, 2),
        }
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

_ASR: dict[str, Any] = {}
_ASR_MODEL = os.environ.get("FACELAB_ASR_MODEL", "CohereLabs/cohere-transcribe-arabic-07-2026")

def _get_asr():
    if "model" not in _ASR:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        _ASR["processor"] = AutoProcessor.from_pretrained(_ASR_MODEL)
        _ASR["model"] = CohereAsrForConditionalGeneration.from_pretrained(
            _ASR_MODEL,
            device_map="auto",
        )
    return _ASR["processor"], _ASR["model"]

@app.post("/transcribe")
def transcribe(req: TranscribeReq):
    if os.environ.get("FACELAB_ASR", "auto").lower() in {"off", "0", "disabled"}:
        raise HTTPException(
            status_code=503,
            detail="ASR is disabled on this backend (FACELAB_ASR=off). Type the prompt instead.",
        )
    if ENGINE.is_mock:
        sample = (
            "رجل في الثلاثينات، جبهة عريضة، عيون ضيقة، لحية كثيفة"
            if req.language == "ar"
            else "a man in his thirties with a broad forehead, narrow eyes, and a thick beard"
        )
        return {"text": sample, "mock": True}

    if len(req.audio) > MAX_DATA_URL_CHARS:
        raise HTTPException(status_code=413, detail="Audio upload is too large.")
    source_path = None
    wav_path = None
    try:
        payload = req.audio.split(",", 1)[1] if req.audio.startswith("data:") else req.audio
        raw = base64.b64decode(payload, validate=True)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as source:
            source.write(raw)
            source_path = source.name
        wav_path = source_path + ".wav"
        conversion = subprocess.run(
            ["ffmpeg", "-y", "-i", source_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True,
            text=True,
        )
        audio_path = wav_path if conversion.returncode == 0 else source_path

        from transformers.audio_utils import load_audio

        audio = load_audio(audio_path, sampling_rate=16000)
        if len(audio) < 1600 or float(np.sqrt(np.mean(np.square(audio)))) < 1e-4:
            raise ValueError("The recording is silent or too short.")
        processor, model = _get_asr()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=req.language)
        inputs = inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return {"text": text, "mock": False, "model": _ASR_MODEL}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc
    finally:
        for path in (source_path, wav_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

class ReconstructReq(BaseModel):
    image: str | None = None
    signature: list[float] | None = None
    occlusion: str | None = Field(default=None, max_length=16)
    mirror: bool = False
    use_faceid: bool = False
    gender: str | None = Field(default=None, max_length=16)
    model: str | None = Field(default=None, pattern="^(dev|klein)$")

class BlendReq(BaseModel):
    images: list[str] = Field(min_length=1, max_length=10)
    gender: str | None = Field(default=None, max_length=16)
    reconstruct: bool = False
    model: str | None = Field(default=None, pattern="^(dev|klein)$")

class HalfFaceReq(BaseModel):
    """Complete a half-face photo, keeping the supplied half pixel-identical."""

    image: str
    side: str | None = Field(default=None, pattern="^(auto|left|right|top|bottom)$")
    # Hand in a COMPLETE photo plus take="left" and the server keeps only that
    # half before completing it, so the discarded half can score the result.
    take: str | None = Field(default=None, pattern="^(left|right|top|bottom)$")
    gender: str | None = Field(default=None, max_length=16)
    prompt: str | None = Field(default=None, max_length=1200)
    model: str | None = Field(default=None, pattern="^(dev|klein)$")
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    tone_match: bool = True
    seam_band: int | None = Field(default=None, ge=0, le=256)
    align_radius: int = Field(default=6, ge=0, le=24)
    align_scale: float = Field(default=0.08, ge=0.0, le=0.3)
    identity_anchor: bool = True
    reference: str | None = None
    # Frontalization route only (profile uploads): total candidate pool, split
    # across the two prompt styles (see faceid_reconstruct2). 8 = 4 seeds x 2.
    candidates: int = Field(default=8, ge=1, le=8)
    refine: bool = False

def _mirror_fill(img: Image.Image, occlusion: str | None) -> Image.Image:
    """Crude symmetric pre-fill: reflect the visible half over the vertical
    midline to cover a left/right occlusion, giving the model a better start."""
    from PIL import ImageOps

    occ = (occlusion or "").strip().lower()
    if occ not in ("left", "right"):
        return img
    w, h = img.size
    mid = w // 2
    out = img.copy()
    if occ == "left":
        half = out.crop((mid, 0, w, h))
        out.paste(ImageOps.mirror(half), (0, 0))
    else:
        half = out.crop((0, 0, mid, h))
        out.paste(ImageOps.mirror(half), (mid, 0))
    return out

_OCC_HINT = {
    "left": "the left side of the face is hidden",
    "right": "the right side of the face is hidden",
    "lower": "the lower face (mouth and chin) is covered",
    "upper": "the upper face is covered",
    "profile": "this is a side/profile angle, not frontal",
    "mask": "the face is partly covered by a mask or covering",
}

def _faceid_prompt(gender: str | None) -> str:
    import faceid_reconstruct as _fid
    return _gender_phrase(gender) + _fid._DEFAULT_PROMPT

def _reconstruct_faceid(embedding: np.ndarray, gender: str | None) -> Image.Image:
    """Paper method: realistic frontal face from the ArcFace embedding via IP-Adapter-FaceID."""
    import faceid_reconstruct as fid
    with MODEL_LOCK:
        return fid.reconstruct(embedding, prompt=_faceid_prompt(gender), seed=7)

def _reconstruct_edit(image: Image.Image, occlusion: str | None, gender: str | None, model: str | None) -> Image.Image:
    """Fallback: identity-preserving FLUX edit that frontalizes/de-occludes the photo."""
    hint = _OCC_HINT.get((occlusion or "").strip().lower(), "the face is partly incomplete")
    instruction = (
        f"{_gender_phrase(gender)}Reconstruct a clear, complete, front-facing passport-style "
        f"portrait of this same person. Note: {hint}. Fill in any covered, cropped, missing, "
        "mirrored or side-angled parts naturally and symmetrically, preserving their identity, "
        "face shape, skin tone, age and distinguishing features. Neutral expression, even "
        "frontal lighting, plain background."
    )
    edit_kwargs: dict[str, Any] = {"mode": "preserve", "seed": 7}
    if getattr(ENGINE, "remote", False) and model:
        edit_kwargs["model"] = model
    with MODEL_LOCK:
        return ENGINE.edit(image, instruction, image, **edit_kwargs)

@app.post("/reconstruct")
def reconstruct(req: ReconstructReq):
    """Turn a partial / side / occluded / masked face (or a bare embedding) into a clean
    frontal portrait + ArcFace signature. With use_faceid=True it uses the paper's method
    (IP-Adapter-FaceID from the embedding) and falls back to the FLUX edit if unavailable."""
    if getattr(ENGINE, "is_mock", False):
        raise fail(RuntimeError("Reconstruction needs the real generation engine (not mock mode)."), 400)
    try:
        image = from_data_url(req.image) if req.image else None
        if image is not None and req.mirror:
            image = _mirror_fill(image, req.occlusion)

        embedding = None
        if req.signature:
            embedding = np.asarray(req.signature, dtype=np.float32).reshape(-1)
        elif image is not None:
            with MODEL_LOCK:
                embedding = require_signature(image, "input face")

        method = None
        out = None
        if req.use_faceid and embedding is not None:
            try:
                out = _reconstruct_faceid(embedding, req.gender)
                method = "faceid"
            except Exception:
                out = None
        if out is None:
            if image is None:
                raise ValueError("An image is required for reconstruction when IP-Adapter-FaceID is unavailable.")
            out = _reconstruct_edit(image, req.occlusion, req.gender, req.model)
            method = "flux-edit"

        with MODEL_LOCK:
            vector = require_signature(out, "reconstructed face")
        return {"image": to_data_url(out, lossless=True), "signature": vec_list(vector),
                "method": method, "mock": ENGINE.is_mock}
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

def _half_face_renderer(req: HalfFaceReq, *, low_guidance: bool = True):
    """Adapter handing half_face whatever generator this deployment has.

    fal.ai FLUX.2 [dev] is preferred and used even when another provider is the
    global default, because it is the one route whose edit endpoint returns the
    canvas at the exact pixel size it was given.
    """
    import half_face as hf

    use_fal = False
    try:
        import fal_provider

        use_fal = fal_provider.available()
    except Exception:
        use_fal = False
    if low_guidance:
        # Half-face wants a much lower guidance than ordinary generation: it keeps
        # the identity closer and stops the model re-cropping. See hf.DEFAULT_GUIDANCE.
        guidance = hf.DEFAULT_GUIDANCE if req.guidance is None else req.guidance
    else:
        # Frontalizing a profile is a big structural change; leave guidance at the
        # provider default unless the caller pinned one.
        guidance = req.guidance

    def render(images, prompt, width, height, seed=None):
        use_seed = req.seed if seed is None else seed
        if use_fal and getattr(ENGINE, "provider", "") != "fal":
            import fal_provider

            w, h = fal_provider.fit_size(width, height)
            out, slug = fal_provider.edit_images(
                prompt, images, width=w, height=h, seed=use_seed,
                steps=req.steps, guidance=guidance, model=req.model,
            )
            return out[0], {"provider": "fal", "model": slug, "render_size": [w, h]}
        with MODEL_LOCK:
            return ENGINE.render_edit(
                images, prompt, width=width, height=height, seed=use_seed,
                steps=req.steps, guidance=guidance, model=req.model,
            )

    return render

def _frontalize_profile(
    req: HalfFaceReq, image: Image.Image, pose: dict[str, Any],
    truth: Image.Image | None = None,
) -> dict[str, Any]:
    """A side-profile photo is a complete head, not half of a frontal portrait —
    mirroring it builds a two-faced canvas. Run the identity-first reconstruction
    instead (faceid_reconstruct2): a prompt-diverse candidate pool ranked against
    the fused profile+mirror identity. No pixel of the result can be guaranteed,
    so ``pixel_exact`` is honestly False here."""
    import faceid_reconstruct2 as f2

    render = _half_face_renderer(req, low_guidance=False)

    embed = None
    if not getattr(ENGINE, "is_mock", False):
        def embed(img):
            with MODEL_LOCK:
                return ENGINE.signature(img)

    # ``candidates`` counts the whole pool; it is split across the two prompt
    # styles (descriptive studio + scene-preserving) measured in the battery.
    per_style = max(1, req.candidates // 2)
    result = f2.reconstruct(
        image,
        renderer=render,
        embed=embed,
        seeds=f2.DEFAULT_SEEDS[:per_style],
        who=_gender_phrase(req.gender).rstrip(", "),
        extra_prompt=req.prompt or "",
    )

    out = result["image"]
    signature = None
    identity = {}
    if embed is not None:
        with MODEL_LOCK:
            signature = ENGINE.signature(out)
            truth_vec = ENGINE.signature(truth) if truth is not None else None
        identity = {
            "vs_supplied_half": result["sim_profile"],
            "vs_reference": ENGINE.similarity(signature, truth_vec),
        }
    # Every candidate travels to the UI so the user can overrule the ArcFace
    # pick with their own eyes. JPEG rather than lossless: 6 full-size PNGs
    # would weigh megabytes and none of these carries a pixel-exact guarantee.
    candidates = [
        {
            **meta,
            "image": to_data_url(img, quality=90),
            "picked": meta["stage"] == result["stage"] and meta["seed"] == result["seed"],
        }
        for meta, img in zip(result["candidates"], result["candidate_images"])
    ]
    return {
        "image": to_data_url(out, lossless=True),
        "half": to_data_url(image, lossless=True),
        "signature": vec_list(signature),
        "pixel_exact": False,
        "method": "frontalized",
        "pose": pose,
        "stage": result["stage"],
        "seed_used": result["seed"],
        "candidates": candidates,
        "identity": identity,
        "render": result["render"],
        "seconds": result["seconds"],
        "mock": getattr(ENGINE, "is_mock", False),
    }

@app.post("/half_face")
def half_face(req: HalfFaceReq):
    """Rebuild a whole face from one half of a photo.

    The half that was supplied is returned byte-for-byte identical — it is
    pasted back over the render at its exact coordinates, and the response
    carries ``pixel_exact`` to say so. Only the missing half is generated.

    Uploads are classified first (``method`` in the response says which route
    ran): a genuine half runs the pixel-exact pipeline; a side-profile photo is
    frontalized instead, because mirroring a whole head yields a two-faced
    canvas; an already-complete frontal photo comes back unchanged.

    ``take`` is the demo/validation path: hand it a complete photo and the
    server keeps only that half before completing it, so the reconstruction can
    be scored against the original it came from.
    """
    import half_face as hf

    try:
        image = from_data_url(req.image)
        truth = None
        if req.take:
            truth = image
            image = hf.take_half(image, req.take)
        elif req.reference:
            truth = from_data_url(req.reference)

        detector = None
        if not getattr(ENGINE, "is_mock", False):
            def detector(img):
                with MODEL_LOCK:
                    return ENGINE.detect_faces(img)

        # A profile photo or an already-complete face must never be mirrored —
        # that is what produced two-headed results. ``take`` skips the check:
        # there the server cut the half itself, so its nature is known.
        if not req.take and detector is not None:
            kind, pose = hf.classify(image, detector)
            if kind == "profile":
                return _frontalize_profile(req, image, pose, truth=truth)
            if kind == "full":
                with MODEL_LOCK:
                    signature = ENGINE.signature(image)
                return {
                    "image": to_data_url(image, lossless=True),
                    "half": to_data_url(image, lossless=True),
                    "signature": vec_list(signature),
                    "pixel_exact": True,
                    "method": "already-complete",
                    "pose": pose,
                    "identity": {},
                    "render": {},
                    "seconds": 0.0,
                    "mock": getattr(ENGINE, "is_mock", False),
                }

        started = time.time()
        result = hf.complete(
            image,
            renderer=_half_face_renderer(req),
            side=req.side,
            gender=_gender_phrase(req.gender).rstrip(", "),
            extra_prompt=req.prompt or "",
            seed=req.seed,
            tone_match=req.tone_match,
            seam_band=req.seam_band,
            align_radius=req.align_radius,
            align_scale=req.align_scale,
            identity_anchor=req.identity_anchor,
            detector=detector,
        )
        elapsed = round(time.time() - started, 2)

        out = result["image"]
        signature = None
        identity = {}
        if not getattr(ENGINE, "is_mock", False):
            with MODEL_LOCK:
                signature = ENGINE.signature(out)
                half_vec = ENGINE.signature(result["half"])
                truth_vec = ENGINE.signature(truth) if truth is not None else None
            identity = {
                "vs_supplied_half": ENGINE.similarity(signature, half_vec),
                "vs_reference": ENGINE.similarity(signature, truth_vec),
            }

        payload = {
            "image": to_data_url(out, lossless=True),
            "half": to_data_url(result["half"], lossless=True),
            "mirror_canvas": to_data_url(result["canvas"], lossless=True),
            "signature": vec_list(signature),
            "method": "half-completion",
            "pixel_exact": result["pixel_exact"],
            "geometry": result["geometry"].as_dict(),
            "side_detected_by": result["side_detected_by"],
            "align": result["align"],
            "align_residual": result["align_residual"],
            "seam_step": result["seam_step"],
            "seam_band": result["seam_band"],
            "identity": identity,
            "render": result["render"],
            "seconds": elapsed,
            "mock": getattr(ENGINE, "is_mock", False),
        }
        if truth is not None:
            payload["accuracy"] = hf.verify(result, truth)
        return payload
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.post("/blend_identity")
def blend_identity(req: BlendReq):
    """Average the ArcFace embeddings of up to 10 photos of ONE person into a single
    robust identity vector (jitter cancels, stable features stay). Optionally synthesize a
    clean frontal face from that blend via IP-Adapter-FaceID (the paper's method)."""
    if getattr(ENGINE, "is_mock", False):
        raise fail(RuntimeError("Blending needs the real ArcFace engine (not mock mode)."), 400)
    try:
        pairs = []
        for data in req.images[:10]:
            img = from_data_url(data)
            with MODEL_LOCK:
                v = ENGINE.signature(img)
            if v is not None:
                pairs.append((img, np.asarray(v, dtype=np.float32).reshape(-1)))
        if not pairs:
            raise ValueError("No detectable face was found in the uploaded images.")
        mat = np.stack([v for _, v in pairs])
        blended = np.mean(mat, axis=0)
        norm = float(np.linalg.norm(blended))
        blended = blended / (norm or 1.0)

        sims = [float(np.dot(v / (np.linalg.norm(v) or 1.0), blended)) for _, v in pairs]
        rep_img = pairs[int(np.argmax(sims))][0]

        out_img = rep_img
        method = "representative"
        if req.reconstruct:
            try:
                out_img = _reconstruct_faceid(blended, req.gender)
                method = "faceid"
            except Exception:
                out_img = rep_img
                method = "representative"

        return {
            "image": to_data_url(out_img, lossless=True),
            "signature": vec_list(blended),
            "used": len(pairs),
            "requested": len(req.images),
            "method": method,
            "mock": ENGINE.is_mock,
        }
    except ValueError as exc:
        raise fail(exc, 422) from exc
    except Exception as exc:
        raise fail(exc) from exc

@app.get("/faceid/status")
def faceid_status():
    try:
        import faceid_reconstruct as fid
        return fid.status()
    except Exception as exc:
        return {"available": False, "error": str(exc)}

@app.get("/gallery/original")
def gallery_original(obj: str = ""):
    """Stream a full-resolution gallery original from MinIO (proxied through the app so
    it works on a closed network where only this service is reachable)."""
    if not obj:
        raise fail(RuntimeError("Missing object name."), 422)
    try:
        import minio_store
        data, content_type = minio_store.get_object(obj)
    except Exception as exc:
        raise fail(RuntimeError(f"Could not fetch original: {exc}"), 404)
    return Response(content=data, media_type=content_type)

_BUILD: dict[str, Any] = {"active": False, "name": None, "done": 0, "total": 0, "kept": 0, "error": None, "finished": False, "started": 0.0, "rate": 0.0, "eta": None}
_BUILD_LOCK = threading.Lock()
_COUNT_CACHE: dict[str, tuple[float, int]] = {}

def _slug(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or "").strip())
    return s[:60] or "gallery"

def _index_count(d: Path) -> int:
    p = d / "front_ids.npy"
    if not p.exists():
        return 0
    key = str(p)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return 0
    cached = _COUNT_CACHE.get(key)
    if cached and cached[0] == mt:
        return cached[1]
    try:
        n = int(len(np.load(p, allow_pickle=True)))
    except Exception:
        n = 0
    _COUNT_CACHE[key] = (mt, n)
    return n

def _gallery_dirs() -> list[dict[str, Any]]:
    out = [{"name": "default", "label": "الافتراضية (المعرض الأساسي)", "count": _index_count(INDEX_DIR), "kind": "faiss"}]
    try:
        import gallery_store as gs
        for c in gs.collections():
            out.append({"name": c["name"], "label": c["name"], "count": c["count"], "kind": "qdrant"})
    except Exception:
        pass
    if _GALLERY_ROOT.exists():
        for d in sorted(_GALLERY_ROOT.iterdir()):
            if d.name == "_uploads":
                continue
            if d.is_dir() and (d / "front.index").exists():
                out.append({"name": d.name, "label": d.name, "count": _index_count(d), "kind": "faiss"})
    return out

def _active_name() -> str:
    if _active_is_qdrant():
        return _ACTIVE_QDRANT
    return "default" if _active_dir() == INDEX_DIR else _active_dir().name

def _find_gallery_images(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() in _BASENAME_EXT or p.suffix.lower() == ""):
            out.append(p)
    return sorted(out)

def _build_gallery(src: Path, out_dir: Path, name: str, thumb: int, limit: int) -> None:
    import faiss

    global _ACTIVE_DIR
    files = _find_gallery_images(src)
    if limit:
        files = files[:limit]
    with _BUILD_LOCK:
        _BUILD.update(active=True, name=name, done=0, total=len(files), kept=0, error=None, finished=False)
    if not files:
        with _BUILD_LOCK:
            _BUILD.update(active=False, finished=True, error="No image files found in the source.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = out_dir / "gallery_thumbs"
    if thumb:
        thumbs.mkdir(parents=True, exist_ok=True)
    vecs: list = []
    ids: list = []
    paths: dict = {}
    kept = 0
    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
            with MODEL_LOCK:
                v = ENGINE.signature(img)
        except Exception:
            v = None
        if v is not None:
            v = np.asarray(v, np.float32).reshape(-1)
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                v = v / nrm
                pid = f"{p.parent.name}/{p.stem}" if p.stem else p.name
                if thumb:
                    t = img.copy(); t.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                    tp = thumbs / f"{kept:07d}.jpg"; t.save(tp, "JPEG", quality=82)
                    paths[pid] = str(tp)
                else:
                    paths[pid] = str(p)
                vecs.append(v); ids.append(pid); kept += 1
        if (i % 10) == 0 or i == len(files) - 1:
            with _BUILD_LOCK:
                _BUILD.update(done=i + 1, kept=kept)
    if kept < 1:
        with _BUILD_LOCK:
            _BUILD.update(active=False, finished=True, error="No faces were detected in any image.")
        return
    mat = np.ascontiguousarray(np.stack(vecs).astype(np.float32))
    idx = faiss.IndexFlatIP(mat.shape[1]); idx.add(mat)
    faiss.write_index(idx, str(out_dir / "front.index"))
    np.save(out_dir / "front_ids.npy", np.array(ids, dtype=object))
    (out_dir / "front_paths.json").write_text(json.dumps(paths), encoding="utf-8")
    _ACTIVE_DIR = out_dir
    _path_map.cache_clear(); _basename_map.cache_clear()
    with _BUILD_LOCK:
        _BUILD.update(active=False, finished=True, done=len(files), kept=kept, name=name)

def _build_gallery_qdrant(src: Path, slug: str, name: str, thumb: int, limit: int) -> None:
    """Embed a folder of images into a Qdrant collection (vector + thumbnail per face)."""
    import gallery_store as gs

    global _ACTIVE_QDRANT
    files = _find_gallery_images(src)
    if limit:
        files = files[:limit]
    started = time.time()
    with _BUILD_LOCK:
        _BUILD.update(active=True, name=name, done=0, total=len(files), kept=0, error=None, finished=False, started=started, rate=0.0, eta=None)
    if not files:
        with _BUILD_LOCK:
            _BUILD.update(active=False, finished=True, error="No image files found in the source.")
        return

    def _emb(img):
        with MODEL_LOCK:
            return ENGINE.signature(img)

    def _prog(done, kept):
        elapsed = time.time() - started
        rate = (done / elapsed) if elapsed > 0 and done > 0 else 0.0
        eta = ((len(files) - done) / rate) if rate > 0 else None
        with _BUILD_LOCK:
            _BUILD.update(done=done, kept=kept, rate=rate, eta=eta)

    kept = gs.build(slug, files, _emb, thumb=thumb, progress=_prog)
    if kept < 1:
        with _BUILD_LOCK:
            _BUILD.update(active=False, finished=True, error="No faces were detected in any image.")
        return
    _ACTIVE_QDRANT = slug
    with _BUILD_LOCK:
        _BUILD.update(active=False, finished=True, done=len(files), kept=kept, name=name)

def _safe_build(src: Path, slug: str, name: str, thumb: int, limit: int) -> None:
    try:
        _build_gallery_qdrant(src, slug, name, thumb, limit)
    except Exception as exc:
        with _BUILD_LOCK:
            _BUILD.update(active=False, finished=True, error=str(exc))

def _start_build(src: Path, name: str, thumb: int, limit: int) -> str:
    with _BUILD_LOCK:
        if _BUILD["active"]:
            raise RuntimeError("A gallery build is already running — wait for it to finish.")
    slug = _slug(name)
    threading.Thread(target=_safe_build, args=(src, slug, name, thumb, limit), daemon=True).start()
    return slug

class GalleryPathReq(BaseModel):
    path: str = Field(min_length=1, max_length=2000)
    name: str = Field(default="", max_length=60)
    thumb: int = Field(default=200, ge=0, le=512)
    limit: int = Field(default=0, ge=0, le=1_000_000)

class GalleryZipReq(BaseModel):
    zip: str
    name: str = Field(default="", max_length=60)
    thumb: int = Field(default=200, ge=0, le=512)
    limit: int = Field(default=0, ge=0, le=1_000_000)

class GalleryActivateReq(BaseModel):
    name: str = Field(min_length=1, max_length=60)

@app.post("/gallery/build_path")
def gallery_build_path(req: GalleryPathReq):
    if getattr(ENGINE, "is_mock", False):
        raise fail(RuntimeError("Building a gallery needs the real ArcFace engine (not mock)."), 400)
    src = Path(req.path).expanduser()
    if not src.exists() or not src.is_dir():
        raise fail(RuntimeError(f"Folder not found on the server: {src}"), 400)
    name = req.name.strip() or src.name
    _start_build(src, name, req.thumb, req.limit)
    return {"ok": True, "name": name, "slug": _slug(name)}

@app.post("/gallery/build_zip")
def gallery_build_zip(req: GalleryZipReq):
    import shutil
    import zipfile

    if getattr(ENGINE, "is_mock", False):
        raise fail(RuntimeError("Building a gallery needs the real ArcFace engine (not mock)."), 400)
    name = req.name.strip() or "uploaded"
    payload = req.zip.split(",", 1)[1] if req.zip.startswith("data:") else req.zip
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise fail(ValueError("Invalid zip upload."), 422) from exc
    up_root = _GALLERY_ROOT / "_uploads" / _slug(name)
    if up_root.exists():
        shutil.rmtree(up_root, ignore_errors=True)
    up_root.mkdir(parents=True, exist_ok=True)
    zpath = up_root / "images.zip"
    zpath.write_bytes(raw)
    try:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(up_root)
    except Exception as exc:
        raise fail(ValueError("Could not read the uploaded zip."), 422) from exc
    _start_build(up_root, name, req.thumb, req.limit)
    return {"ok": True, "name": name, "slug": _slug(name)}

@app.get("/gallery/status")
def gallery_status():
    with _BUILD_LOCK:
        build = dict(_BUILD)
    return {"build": build, "active": _active_name(), "galleries": _gallery_dirs(), "ready": search_status()["ready"]}

@app.get("/gallery/list")
def gallery_list():
    return {"active": _active_name(), "galleries": _gallery_dirs()}

@app.post("/gallery/activate")
def gallery_activate(req: GalleryActivateReq):
    global _ACTIVE_DIR, _ACTIVE_QDRANT
    name = req.name.strip()
    if name in ("default", "افتراضية", ""):
        _ACTIVE_DIR = INDEX_DIR
        _ACTIVE_QDRANT = None
        _path_map.cache_clear(); _basename_map.cache_clear()
        return {"ok": True, "active": "default", "kind": "faiss", "count": _index_count(INDEX_DIR)}
    slug = _slug(name)
    try:
        import gallery_store as gs
        if gs.exists(slug):
            _ACTIVE_QDRANT = slug
            return {"ok": True, "active": name, "kind": "qdrant", "count": gs.count(slug)}
    except Exception:
        pass
    target = _GALLERY_ROOT / slug
    if (target / "front.index").exists():
        _ACTIVE_DIR = target
        _ACTIVE_QDRANT = None
        _path_map.cache_clear(); _basename_map.cache_clear()
        return {"ok": True, "active": name, "kind": "faiss", "count": _index_count(target)}
    raise fail(RuntimeError(f"Gallery '{name}' has no index yet."), 404)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
