"""bench_newdataset — paired-set benchmark for the New DataSet goal (2026-08-10).

Runs the production-style pipeline (evidence build → VLM examiner description →
forensic/std prompt → ONE fal render) on every PAIRED sample produced by
classify_new_dataset.py, then scores generated vs TRUE frontal:

  arcface_cos   buffalo_l cosine similarity (headline metric)
  landmark_nme  106-pt landmark error after 5-pt similarity alignment of each
                face to the 112px ArcFace template, mean L2 / 112 (lower=better)

Budget discipline: ONE render per sample per run (fal balance is capped at $2
for this whole campaign). VLM descriptions are cached to descriptions.json and
reused across runs so iterations change exactly one variable. --mock renders
nothing and costs nothing.

Outputs per run: Test_AI_Half/NewDataSet_Bench/<tag>/
  <key>__gen.png, <key>__board.png, report.json
and appends one summary row to Test_AI_Half/NewDataSet_Bench/results_table.md

Usage (from Forensic_Art_v8):
  python tools/bench_newdataset.py --model dev --seed 42 [--mock] [--only key]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import fal_provider  # noqa: E402
from halfface import build_evidence  # noqa: E402
from halfface import describe, prompts  # noqa: E402
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402

SPLIT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "NewDataSet_Split"
BENCH = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "NewDataSet_Bench"
DESC_CACHE = BENCH / "descriptions.json"

# rough fal pricing used only for the spend log (measured: ~47 flux renders ≈ $2.82)
EST_PRICE = {"dev": 0.06, "qwen": 0.03, "klein": 0.03}

# ArcFace 112×112 5-point template (insightface arcface_dst)
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def biggest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) \
        if faces else None


def landmarks_112(face) -> "np.ndarray | None":
    """Map the 106-pt landmarks into ArcFace 112px template space via the
    similarity transform fitted on the 5 kps — pose/scale-free comparison."""
    import cv2

    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is None:
        return None
    M, _ = cv2.estimateAffinePartial2D(np.asarray(face.kps, dtype=np.float32),
                                       ARC_DST, method=cv2.LMEDS)
    if M is None:
        return None
    pts = np.asarray(lmk, dtype=np.float32)
    return pts @ M[:, :2].T + M[:, 2]


def landmark_nme(gen: Image.Image, truth: Image.Image, detector) -> "float | None":
    fg, ft = biggest_face(detector(gen)), biggest_face(detector(truth))
    if fg is None or ft is None:
        return None
    a, b = landmarks_112(fg), landmarks_112(ft)
    if a is None or b is None:
        return None
    return round(float(np.mean(np.linalg.norm(a - b, axis=1))) / 112.0, 4)


def load_pairs() -> "list[tuple[str, Path, Path]]":
    pairs = []
    for inp in sorted((SPLIT / "paired").glob("*__input.png")):
        key = inp.name[: -len("__input.png")]
        tgt = SPLIT / "paired" / f"{key}__target.png"
        if tgt.exists():
            pairs.append((key, inp, tgt))
    return pairs


def get_description(key: str, ev, cache: dict) -> str:
    if key in cache:
        return cache[key]["description"]
    desc, meta = describe.from_image(ev)
    cache[key] = {"description": desc, "meta": {k: v for k, v in meta.items()
                                               if isinstance(v, (str, int, float, bool))}}
    DESC_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return desc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dev", help="fal edit model short name (dev|qwen|klein)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--style", default="forensic")
    ap.add_argument("--tag", default=None, help="output folder name; default <model>-s<seed>")
    ap.add_argument("--note", default="", help="what changed vs baseline (for the results table)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", action="append", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model}-{args.style}-s{args.seed}" + ("-MOCK" if args.mock else "")

    out = BENCH / tag
    out.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    cache = json.loads(DESC_CACHE.read_text(encoding="utf-8")) if DESC_CACHE.exists() else {}

    pairs = load_pairs()
    if args.only:
        pairs = [p for p in pairs if p[0] in set(args.only)]
    print(f"[bench] {len(pairs)} paired samples, model={args.model}, seed={args.seed}, "
          f"style={args.style}, mock={args.mock}", flush=True)

    rows: dict[str, dict] = {}
    renders = 0
    for key, inp_path, tgt_path in pairs:
        t0 = time.time()
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        ev = build_evidence(inp, None, detector=detector, embed=embed)
        desc = "" if args.mock and key not in cache else get_description(key, ev, cache)
        prompt = prompts.build(args.style, description=desc, n_photos=len(ev.refs_std))
        w, h = fal_provider.fit_size(ev.working.width, ev.working.height)

        try:
            if args.mock:
                gen = ev.working.resize((w, h), Image.Resampling.LANCZOS)
            else:
                gen = fal_provider.edit_images(prompt, ev.refs_std, width=w, height=h,
                                               seed=args.seed, model=args.model)[0][0]
                renders += 1
        except Exception as exc:
            print(f"[{key}] RENDER FAILED: {exc}", flush=True)
            if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                raise SystemExit(f"ABORT — fal balance problem: {exc}")
            rows[key] = {"error": str(exc)[:200]}
            continue

        cosv = vs_image(gen, tgt, embed)
        nme = landmark_nme(gen, tgt, detector)
        rows[key] = {"arcface_cos": cosv, "landmark_nme": nme,
                     "desc_chars": len(desc), "tier": ev.tier,
                     "n_refs": len(ev.refs_std), "secs": round(time.time() - t0, 1)}
        gen.save(out / f"{key}__gen.png")
        label_strip([("PROFILE INPUT", inp), (f"GENERATED cos={cosv} nme={nme}", gen),
                     ("TRUE FRONTAL", tgt)], height=460).save(out / f"{key}__board.png")
        print(f"[{key}] cos={cosv} nme={nme} tier={ev.tier} desc={len(desc)}ch "
              f"({rows[key]['secs']}s)", flush=True)

    scored = [r for r in rows.values() if r.get("arcface_cos") is not None]
    coss = sorted(r["arcface_cos"] for r in scored)
    nmes = sorted(r["landmark_nme"] for r in scored if r.get("landmark_nme") is not None)
    worst_key = min((k for k, r in rows.items() if r.get("arcface_cos") is not None),
                    key=lambda k: rows[k]["arcface_cos"], default=None)
    est = round(renders * EST_PRICE.get(args.model, 0.06), 2)

    def stats(v):
        return {"mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4),
                "worst": round(float(v[0]), 4)} if v else None

    summary = {"tag": tag, "model": args.model, "seed": args.seed, "style": args.style,
               "note": args.note, "n": len(scored), "renders": renders,
               "est_cost_usd": est, "mock": args.mock,
               "arcface_cos": stats(coss),
               "landmark_nme": (stats(sorted(nmes, reverse=True)) if nmes else None),
               "worst_sample": worst_key,
               "below_0.4": [k for k, r in rows.items()
                             if (r.get("arcface_cos") or 1) < 0.4]}
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")

    if not BENCH.joinpath("results_table.md").exists():
        BENCH.joinpath("results_table.md").write_text(
            "| run | note | n | cos mean | cos median | cos worst | nme mean | renders | est $ |\n"
            "|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    ac, nm = summary["arcface_cos"], summary["landmark_nme"]
    with BENCH.joinpath("results_table.md").open("a", encoding="utf-8") as f:
        f.write(f"| {tag} | {args.note} | {summary['n']} | "
                f"{ac['mean'] if ac else '—'} | {ac['median'] if ac else '—'} | "
                f"{ac['worst'] if ac else '—'} | "
                f"{(round(float(np.mean(nmes)),4)) if nmes else '—'} | {renders} | {est} |\n")

    print("\nSUMMARY " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
