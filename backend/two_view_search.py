
import os
import glob
import numpy as np
from collections import defaultdict

from face_pipeline import (
    _get_face_app, _embed_largest_face,
    generate_faces, reduce_to_query,
)

def inspect_dataset(root: str, n: int = 40):
    """Print a sample of the tree so you can confirm the front/side layout
    BEFORE indexing. Run this first; adjust FRONT_KEY/SIDE_KEY below if needed."""
    files = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            files.append(os.path.join(dp, f))
    print(f"{len(files)} files under {root}")
    for p in files[:n]:
        print("  ", os.path.relpath(p, root))
    return files

FRONT_KEY = "front"
SIDE_KEY = "side"

def _person_id(path: str) -> str:
    """Person id = filename with the view token and extension stripped.
    The front and side image of the same person must map to the SAME id, so we
    remove the view keyword. Adjust if the dataset pairs by folder index instead."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for tok in (FRONT_KEY, SIDE_KEY, "_f", "_s", "-f", "-s"):
        stem = stem.replace(tok, "")
    return stem.strip("_-. ")

def split_front_side(root: str):
    """Return paired front/side maps keyed by the shared person id.

    Match directory components exactly instead of searching for the words
    anywhere in a full path (which can misclassify parent directory names).
    """
    front_map, side_map = {}, {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext and ext not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        parts = {part.lower() for part in os.path.normpath(p).split(os.sep)}
        pid = _person_id(p)
        if FRONT_KEY in parts:
            front_map[pid] = p
        elif SIDE_KEY in parts:
            side_map[pid] = p
    common = set(front_map) & set(side_map)
    print(f"front={len(front_map)}  side={len(side_map)}  paired={len(common)}")
    return front_map, side_map

def build_view_index(id_to_path: dict, index_path: str, ids_path: str,
                     report_every: int = 5000):
    """Embed every image of one view into an exact cosine (IndexFlatIP) index.
    Faces that fail detection (more common on profiles) are skipped and logged."""
    import faiss
    from PIL import Image
    app = _get_face_app()

    index = faiss.IndexFlatIP(512)
    kept_ids, skipped = [], 0
    for i, (pid, path) in enumerate(id_to_path.items()):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        emb = _embed_largest_face(app, img)
        if emb is None:
            skipped += 1
            continue
        index.add(emb[None, :])
        kept_ids.append(pid)
        if (i + 1) % report_every == 0:
            print(f"  {i + 1}/{len(id_to_path)} indexed={index.ntotal} skipped={skipped}")

    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(kept_ids))
    print(f"[{index_path}] indexed {index.ntotal}, skipped {skipped}")
    return index_path, ids_path

def build_both(root: str,
               front_index="front.index", front_ids="front_ids.npy",
               side_index="side.index",  side_ids="side_ids.npy"):
    front_map, side_map = split_front_side(root)
    build_view_index(front_map, front_index, front_ids)
    build_view_index(side_map,  side_index,  side_ids)
    return (front_index, front_ids, side_index, side_ids)

def build_query_views(front_prompt: str, side_prompt: str | None = None,
                      n: int = 30, reduce: str = "mean"):
    """Generate n frontal + n profile faces, reduce each set to ONE robust vector.
    (Averaging is done in embedding space, per the earlier fix.)"""
    if side_prompt is None:
        side_prompt = front_prompt + ", side profile view, head turned 90 degrees, facing left"
    front_imgs = generate_faces(front_prompt, n=n)
    side_imgs  = generate_faces(side_prompt,  n=n)
    front_q, _, nf = reduce_to_query(front_imgs, use=reduce)
    side_q,  _, ns = reduce_to_query(side_imgs,  use=reduce)
    print(f"query built from {nf} frontal + {ns} profile faces")
    return front_q, side_q

def _search(index_path, ids_path, query, topn):
    import faiss
    index = faiss.read_index(index_path)
    ids = np.load(ids_path, allow_pickle=True)
    topn = min(topn, index.ntotal)
    q = np.ascontiguousarray(query[None, :], dtype=np.float32)
    sims, idx = index.search(q, topn)
    return [(str(ids[i]), float(s)) for s, i in zip(sims[0], idx[0]) if i != -1]

def rrf_fuse(rankings, weights=None, k: int = 60):
    """Reciprocal Rank Fusion. rankings: list of best-first [(id, score), ...].
    Being high in BOTH views scores higher than being high in only one."""
    weights = weights or [1.0] * len(rankings)
    fused = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, (pid, _s) in enumerate(ranking):
            fused[pid] += w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

def two_view_search(front_q, side_q,
                    front_index="front.index", front_ids="front_ids.npy",
                    side_index="side.index",  side_ids="side_ids.npy",
                    k_final: int = 3000, mode: str = "rrf",
                    side_weight: float = 0.7, intersect_depth: int = 5000):
    """
    Return the top-k_final person ids by cross-view consensus.

    mode="rrf"       : Reciprocal Rank Fusion (recommended). side_weight < 1.0
                       down-weights the noisier profile view.
    mode="intersect" : your literal rule - keep only ids in BOTH top-`intersect_depth`
                       lists, ranked by combined rank. Brittle but faithful.
    mode="sum"       : sum of cosine similarities across the two views.
    """

    front_rank = _search(front_index, front_ids, front_q, topn=10**9)
    side_rank  = _search(side_index,  side_ids,  side_q,  topn=10**9)

    if mode == "rrf":
        fused = rrf_fuse([front_rank, side_rank], weights=[1.0, side_weight])
        return fused[:k_final]

    if mode == "sum":
        sf = dict(front_rank); ss = dict(side_rank)
        allids = set(sf) | set(ss)
        fused = sorted(((p, sf.get(p, 0.0) + side_weight * ss.get(p, 0.0))
                        for p in allids), key=lambda kv: kv[1], reverse=True)
        return fused[:k_final]

    if mode == "intersect":
        fr = {p: r for r, (p, _) in enumerate(front_rank[:intersect_depth])}
        sr = {p: r for r, (p, _) in enumerate(side_rank[:intersect_depth])}
        both = set(fr) & set(sr)
        ranked = sorted(both, key=lambda p: fr[p] + sr[p])
        return [(p, -(fr[p] + sr[p])) for p in ranked][:k_final]

    raise ValueError(f"unknown mode: {mode}")

if __name__ == "__main__":
    import kagglehub
    root = kagglehub.dataset_download("elliotp/idoc-mugshots")
    print("dataset at:", root)

    inspect_dataset(root)
    build_both(root)

    front_q, side_q = build_query_views(
        "a 30 year old man, short dark hair, broad nose, light stubble", n=30)

    results = two_view_search(front_q, side_q, k_final=3000, mode="rrf",
                              side_weight=0.7)
    print(f"\ntop {len(results)} by front+side consensus:")
    for rank, (pid, score) in enumerate(results[:25], 1):
        print(f"{rank:3d}. id={pid}  fused={score:.5f}")
