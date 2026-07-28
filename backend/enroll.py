"""Add and remove faces in the two-view gallery index."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

PREFIX = "ME::"
FILES = {
    "front_index": "front.index",
    "front_ids": "front_ids.npy",
    "side_index": "side.index",
    "side_ids": "side_ids.npy",
    "front_paths": "front_paths.json",
    "manifest": "enrolled.json",
}

def _p(index_dir: Path, key: str) -> Path:
    return index_dir / FILES[key]

def gallery_ready(index_dir: Path) -> bool:

    return all(_p(index_dir, k).exists() for k in ("front_index", "front_ids", "front_paths"))

def _has_side(index_dir: Path) -> bool:
    return _p(index_dir, "side_index").exists() and _p(index_dir, "side_ids").exists()

def _load_manifest(index_dir: Path) -> list[dict[str, Any]]:
    path = _p(index_dir, "manifest")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_manifest(index_dir: Path, entries: list[dict[str, Any]]) -> None:
    _p(index_dir, "manifest").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

def enrolled_ids(index_dir: Path) -> set[str]:
    return {e["id"] for e in _load_manifest(index_dir)}

def status(index_dir: Path) -> dict[str, Any]:
    entries = _load_manifest(index_dir)
    return {
        "enrolled": len(entries),
        "labels": [e.get("label", "") for e in entries],
        "ids": [e["id"] for e in entries],
        "gallery_ready": gallery_ready(index_dir),
    }

def _append_rows(index_path: Path, ids_path: Path, vectors: np.ndarray, pid: str) -> int:
    """Append `vectors` (rows) to an IndexFlatIP and its id array under one pid."""
    import faiss

    index = faiss.read_index(str(index_path))
    ids = list(np.load(ids_path, allow_pickle=True))
    vectors = np.ascontiguousarray(vectors.astype(np.float32))

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    index.add(vectors)
    ids.extend([pid] * vectors.shape[0])
    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.array(ids, dtype=object))
    return vectors.shape[0]

def enroll(
    index_dir: Path,
    embed: Callable[[Image.Image], "np.ndarray | None"],
    front_images: list[Image.Image],
    side_images: list[Image.Image] | None,
    label: str,
) -> dict[str, Any]:
    """Add one identity (possibly several photos) to both view galleries.

    embed: a callable returning a 512-d L2-normalized ArcFace vector (or None).
    """
    index_dir = Path(index_dir)
    if not gallery_ready(index_dir):
        raise RuntimeError("The 70k gallery is not present; restore the FAISS indexes first (Step 5).")
    if not front_images:
        raise ValueError("At least one frontal photo is required.")

    label = (label or "ME").strip()[:60] or "ME"
    pid = f"{PREFIX}{label}"

    front_vecs = [v for v in (embed(im) for im in front_images) if v is not None]
    if not front_vecs:
        raise ValueError("No face was detected in the frontal photo(s).")
    front_mat = np.stack([np.asarray(v, np.float32).reshape(-1) for v in front_vecs])

    side_source = side_images if side_images else None
    if side_source:
        side_vecs = [v for v in (embed(im) for im in side_source) if v is not None]
        if not side_vecs:
            side_vecs = list(front_vecs)
            reused = True
        else:
            reused = False
    else:
        side_vecs = list(front_vecs)
        reused = True
    side_mat = np.stack([np.asarray(v, np.float32).reshape(-1) for v in side_vecs])

    if pid in enrolled_ids(index_dir):
        clear(index_dir, only=pid)

    front_added = _append_rows(_p(index_dir, "front_index"), _p(index_dir, "front_ids"), front_mat, pid)

    side_added = 0
    if _has_side(index_dir):
        side_added = _append_rows(_p(index_dir, "side_index"), _p(index_dir, "side_ids"), side_mat, pid)

    enrolled_dir = index_dir / "enrolled"
    enrolled_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in label) or "ME"
    image_path = enrolled_dir / f"{safe}.png"
    front_images[0].convert("RGB").save(image_path)

    paths_path = _p(index_dir, "front_paths")
    paths = json.loads(paths_path.read_text(encoding="utf-8"))
    paths[pid] = str(image_path)
    paths_path.write_text(json.dumps(paths), encoding="utf-8")

    entries = [e for e in _load_manifest(index_dir) if e["id"] != pid]
    entries.append({
        "id": pid,
        "label": label,
        "front_rows": front_added,
        "side_rows": side_added,
        "side_reused_front": reused,
        "image_path": str(image_path),
        "created_at": time.time(),
    })
    _save_manifest(index_dir, entries)

    return {
        "id": pid,
        "label": label,
        "front_added": front_added,
        "side_added": side_added,
        "side_reused_front": reused,
        "total_enrolled": len(entries),
    }

def _rebuild_without(index_path: Path, ids_path: Path, drop: set[str]) -> int:
    """Rebuild an IndexFlatIP dropping every row whose id is in `drop`."""
    import faiss

    index = faiss.read_index(str(index_path))
    ids = np.load(ids_path, allow_pickle=True)
    total = index.ntotal
    if total == 0:
        return 0
    keep = [i for i in range(total) if str(ids[i]) not in drop]
    if len(keep) == total:
        return 0
    vectors = index.reconstruct_n(0, total)
    new_index = faiss.IndexFlatIP(vectors.shape[1])
    if keep:
        new_index.add(np.ascontiguousarray(vectors[keep].astype(np.float32)))
    faiss.write_index(new_index, str(index_path))
    np.save(ids_path, np.array([ids[i] for i in keep], dtype=object))
    return total - len(keep)

def clear(index_dir: Path, only: str | None = None) -> dict[str, Any]:
    """Remove enrolled identities from both indexes (all, or a single id)."""
    index_dir = Path(index_dir)
    if not gallery_ready(index_dir):
        return {"removed": 0, "front_removed": 0, "side_removed": 0}
    entries = _load_manifest(index_dir)
    drop = {only} if only else {e["id"] for e in entries}
    if not drop:
        return {"removed": 0, "front_removed": 0, "side_removed": 0}

    front_removed = _rebuild_without(_p(index_dir, "front_index"), _p(index_dir, "front_ids"), drop)
    side_removed = 0
    if _has_side(index_dir):
        side_removed = _rebuild_without(_p(index_dir, "side_index"), _p(index_dir, "side_ids"), drop)

    paths_path = _p(index_dir, "front_paths")
    paths = json.loads(paths_path.read_text(encoding="utf-8"))
    for pid in drop:
        paths.pop(pid, None)
    paths_path.write_text(json.dumps(paths), encoding="utf-8")

    remaining = [e for e in entries if e["id"] not in drop]
    _save_manifest(index_dir, remaining)
    return {
        "removed": len(drop),
        "front_removed": front_removed,
        "side_removed": side_removed,
        "remaining": len(remaining),
    }
