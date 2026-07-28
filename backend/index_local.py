"""Build the front-only search index from a local image dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp",
           ".tif", ".tiff", ".gif", ".ppm", ".pgm"}

def _resolve_src(args) -> Path:
    if args.zip:
        zp = Path(args.zip).expanduser()
        if not zp.exists():
            sys.exit(f"Zip not found: {zp}")
        out = zp.with_suffix("")
        if not out.exists() or not any(out.iterdir()):
            print(f"Extracting {zp.name} → {out} (one time) …")
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                z.extractall(out)
        base = out
    else:
        base = Path(args.src).expanduser()
    if not base.exists():
        sys.exit(f"Source not found: {base}")
    return base

def _find_images(root: Path) -> list[Path]:

    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in IMG_EXT or suf == "":
            out.append(p)
    return sorted(out)

def _diagnose(base: Path) -> None:
    print("\n  No images found. Directory contents:")
    print(f"  Base folder: {base}")
    try:
        subs = [p.name for p in base.iterdir() if p.is_dir()]
        print(f"  Subfolders ({len(subs)}): {subs[:25]}")
    except Exception:
        pass
    exts: dict[str, int] = {}
    samples: list[str] = []
    for i, p in enumerate(base.rglob("*")):
        if i > 60000:
            break
        if p.is_file():
            exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
            if len(samples) < 12:
                samples.append(str(p.relative_to(base)))
    top = sorted(exts.items(), key=lambda x: -x[1])[:15]
    print(f"  File extensions seen (count): {top}")
    print(f"  Sample files: {samples}")
    print("  → Point --src at the exact folder that holds the images.")

def _face_app(use_gpu: bool, det: int = 320):
    import warnings
    warnings.filterwarnings("ignore")
    from insightface.app import FaceAnalysis
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                 else ["CPUExecutionProvider"])

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det, det))
    return app

_WAPP = None

def _worker_init(use_gpu: bool, det: int) -> None:
    global _WAPP
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    _WAPP = _face_app(use_gpu, det)

def _worker_embed(task):
    """task = (idx, path_str, thumb_path_or_'', thumb_size). Returns (idx, pid|None, vec|None, disp|None)."""
    global _WAPP
    from PIL import Image
    idx, path_str, thumb_path, thumb = task
    try:
        img = Image.open(path_str).convert("RGB")
    except Exception:
        return (idx, None, None, None)
    try:
        faces = _WAPP.get(np.asarray(img)[:, :, ::-1])
    except Exception:
        return (idx, None, None, None)
    if not faces:
        return (idx, None, None, None)
    f = max(faces, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
    v = np.asarray(f.normed_embedding, dtype=np.float32)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0:
        return (idx, None, None, None)
    v = v / nrm
    p = Path(path_str)
    pid = f"{p.parent.name}/{p.stem}"
    disp = path_str
    if thumb and thumb_path:
        try:
            t = img.copy(); t.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            t.save(thumb_path, "JPEG", quality=82); disp = thumb_path
        except Exception:
            disp = path_str
    return (idx, pid, [float(x) for x in v], disp)

def _write_index(index_dir: Path, vectors, ids, paths, thumbs_dir, thumb) -> None:
    import faiss
    if len(ids) < 2:
        sys.exit("Too few faces embedded — check the source folder.")
    mat = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(index_dir / "front.index"))
    np.save(index_dir / "front_ids.npy", np.array(ids, dtype=object))
    clean = {k: v for k, v in paths.items() if not k.startswith("_")}
    (index_dir / "front_paths.json").write_text(json.dumps(clean), encoding="utf-8")
    for f in (index_dir / "_index_ckpt.npz", index_dir / "_index_done.json"):
        try: f.unlink()
        except Exception: pass
    print(f"✅ Built front-only gallery: {len(ids):,} faces → {index_dir}")
    if thumb:
        print(f"   Thumbnails in {thumbs_dir}. Whole gallery is small + offline-ready.")
    print("   Restart run_local and search is live (front-only, cosine %).")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="")
    ap.add_argument("--zip", default="")
    ap.add_argument("--front-name", default="front")
    ap.add_argument("--index-dir", default=os.environ.get("FACELAB_INDEX_DIR", "../data"))
    ap.add_argument("--thumb", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--det", type=int, default=320,
                    help="Face-detection window (px). 320 is fast + fine for mugshots; raise to 640 for tiny faces.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel CPU processes (0 = auto = cores-1, capped at 6). 1 = serial.")
    args = ap.parse_args()
    if not args.src and not args.zip:
        sys.exit("Give --src <folder> or --zip <file>. See --help.")

    from PIL import Image
    import faiss

    src = _resolve_src(args)
    index_dir = Path(args.index_dir).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = index_dir / "gallery_thumbs"
    if args.thumb:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
    ckpt = index_dir / "_index_ckpt.npz"
    done_file = index_dir / "_index_done.json"

    front = src / args.front_name
    scope = f"{args.front_name}/"
    print(f"Scanning {src} for images …")
    files = _find_images(front) if front.is_dir() else []
    if not files:
        files = _find_images(src)
        scope = "all folders"
    if not files:
        _diagnose(src)
        sys.exit(1)
    if args.limit:
        files = files[: args.limit]
    print(f"  {len(files):,} images found under {scope}.")

    workers = args.workers if args.workers > 0 else min(6, max(1, (os.cpu_count() or 2) - 1))

    if workers > 1:
        import time
        from concurrent.futures import ProcessPoolExecutor

        ckpt = index_dir / "_index_ckpt.npz"
        done_file = index_dir / "_index_done.json"
        vectors: list = []
        ids: list = []
        paths: dict = {}
        done_idx: set = set()
        if ckpt.exists() and done_file.exists():
            try:
                pj = json.loads(done_file.read_text(encoding="utf-8"))
                if "_done_idx" in pj:
                    data = np.load(ckpt, allow_pickle=True)
                    vectors = list(data["vecs"]); ids = list(data["ids"])
                    paths = pj; done_idx = set(pj["_done_idx"])
                    print(f"  Resuming — {len(done_idx):,} already processed.")
                else:
                    print("  Ignoring an incompatible old checkpoint; starting fresh.")
            except Exception:
                vectors, ids, paths, done_idx = [], [], {}, set()

        tasks = [
            (i, str(files[i]), str(thumbs_dir / f"{i:07d}.jpg") if args.thumb else "", args.thumb)
            for i in range(len(files)) if i not in done_idx
        ]
        print(f"  Embedding with {workers} parallel workers (det {args.det}px)…")
        t0 = time.time(); processed = 0

        def _save():
            paths["_done_idx"] = list(done_idx)
            np.savez(ckpt, vecs=np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 512), np.float32),
                     ids=np.array(ids, dtype=object))
            done_file.write_text(json.dumps(paths), encoding="utf-8")

        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(args.gpu, args.det)) as ex:
                for idx, pid, vec, disp in ex.map(_worker_embed, tasks, chunksize=16):
                    done_idx.add(idx); processed += 1
                    if vec is not None:
                        ids.append(pid); vectors.append(vec); paths[pid] = disp
                    if processed % 200 == 0:
                        rate = processed / max(1e-6, time.time() - t0)
                        eta = (len(tasks) - processed) / max(1e-6, rate)
                        print(f"\r  {processed:,}/{len(tasks):,}  kept {len(ids):,}  ~{rate:.0f}/s  ETA {eta/60:.0f}m",
                              end="", flush=True)
                    if processed % 4000 == 0:
                        _save()
        except Exception as exc:
            _save()
            print(f"\n  Parallel run interrupted ({exc}). Progress saved — re-run to resume.")
            sys.exit(1)
        print(f"\r  embedded {len(ids):,} faces from {len(files):,} images.                    ")
        _write_index(index_dir, vectors, ids, paths, thumbs_dir, args.thumb)
        return

    processed: dict[str, str] = {}
    vectors: list[list[float]] = []
    ids: list[str] = []
    paths: dict[str, str] = {}
    start = 0
    if ckpt.exists() and done_file.exists():
        try:
            data = np.load(ckpt, allow_pickle=True)
            vectors = list(data["vecs"])
            ids = list(data["ids"])
            paths = json.loads(done_file.read_text(encoding="utf-8"))
            processed = {p: "1" for p in paths.get("_processed", [])}
            start = len(processed)
            print(f"  Resuming — {start:,} already embedded.")
        except Exception:
            vectors, ids, paths, processed, start = [], [], {}, {}, 0

    app = _face_app(args.gpu, args.det)
    kept = len(ids)
    seen = set(processed)
    proc_list = list(processed)

    def save_ckpt():
        np.savez(ckpt, vecs=np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 512), np.float32),
                 ids=np.array(ids, dtype=object))
        paths["_processed"] = proc_list
        done_file.write_text(json.dumps(paths), encoding="utf-8")

    import time
    t0 = time.time()
    for i, path in enumerate(files):
        key = str(path)
        if key in seen:
            continue
        if i % 100 == 0:
            rate = kept / max(1e-6, time.time() - t0)
            eta = (len(files) - i) / max(1e-6, rate) if rate else 0
            print(f"\r  {i:,}/{len(files):,}  kept {kept:,}  ~{rate:.0f}/s  ETA {eta/60:.0f}m", end="", flush=True)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            seen.add(key); proc_list.append(key); continue
        faces = app.get(np.asarray(img)[:, :, ::-1])
        seen.add(key); proc_list.append(key)
        if not faces:
            continue
        face = max(faces, key=lambda f: float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        vec = np.asarray(face.normed_embedding, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n <= 0:
            continue
        vec = vec / n
        pid = f"{path.parent.name}/{path.stem}"
        if args.thumb:
            t = img.copy(); t.thumbnail((args.thumb, args.thumb), Image.Resampling.LANCZOS)
            tp = thumbs_dir / f"{kept:06d}.jpg"; t.save(tp, "JPEG", quality=82)
            paths[pid] = str(tp)
        else:
            paths[pid] = str(path)
        vectors.append([float(x) for x in vec]); ids.append(pid); kept += 1
        if kept % 2000 == 0:
            save_ckpt()
    print(f"\r  embedded {kept:,} faces from {len(files):,} images.                    ")

    if kept < 2:
        sys.exit("Too few faces embedded — check the source folder.")

    mat = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(index_dir / "front.index"))
    np.save(index_dir / "front_ids.npy", np.array(ids, dtype=object))
    clean = {k: v for k, v in paths.items() if k != "_processed"}
    (index_dir / "front_paths.json").write_text(json.dumps(clean), encoding="utf-8")

    for f in (ckpt, done_file):
        try: f.unlink()
        except Exception: pass
    print(f"✅ Built front-only gallery: {kept:,} faces → {index_dir}")
    if args.thumb:
        print(f"   Thumbnails in {thumbs_dir}. Whole gallery is small + offline-ready.")
    print("   Restart run_local and search is live (front-only, cosine %).")

if __name__ == "__main__":
    main()
