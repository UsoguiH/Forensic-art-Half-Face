"""iter4_multiphoto — evidence fusion for pairwise-VERIFIED same-subject clusters.

One variable vs ITER1 (qwen, seed 42, cached descriptions): evidence goes from
single profile → profile + other verified cluster members' PROFILE panels
(truth-truth cluster verification >= 0.55 for every pair; frontal evidence like
original_6 excluded — it IS young_6's truth photo, 0.969).

Renders 6 samples x 1 qwen seed. Output folder matches bench format so
pick_newdataset.py can combine runs.

Usage (from Forensic_Art_v8):  python tools/iter4_multiphoto.py
"""
from __future__ import annotations

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
from halfface import build_evidence, prompts  # noqa: E402
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from bench_newdataset import (BENCH, DESC_CACHE, EST_PRICE, landmark_nme,  # noqa: E402
                              load_pairs)

TAG = "ITER4-multiphoto"
SEED = 42

# verified clusters (truth-truth >= 0.55 pairwise, 2026-08-10):
CLUSTERS = {
    "man_1": ["man_3_far_picture"],
    "man_3_far_picture": ["man_1"],
    "young_1": ["young_6", "young_7", "young_near"],
    "young_6": ["young_1", "young_7", "young_near"],
    "young_7": ["young_1", "young_6", "young_near"],
    "young_near": ["young_1", "young_6", "young_7"],
}


def main() -> None:
    out = BENCH / TAG
    out.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    cache = json.loads(DESC_CACHE.read_text(encoding="utf-8"))
    pairs = {k: (i, t) for k, i, t in load_pairs()}

    rows: dict[str, dict] = {}
    renders = 0
    for key, extra_keys in CLUSTERS.items():
        t0 = time.time()
        inp = Image.open(pairs[key][0]).convert("RGB")
        tgt = Image.open(pairs[key][1]).convert("RGB")
        extras = [Image.open(pairs[k][0]).convert("RGB") for k in extra_keys]
        ev = build_evidence(inp, extras, detector=detector, embed=embed)
        desc = cache[key]["description"]
        prompt = prompts.build("forensic", description=desc,
                               n_photos=len(ev.refs_std))
        w, h = fal_provider.fit_size(ev.working.width, ev.working.height)
        try:
            gen = fal_provider.edit_images(prompt, ev.refs_std, width=w, height=h,
                                           seed=SEED, model="qwen")[0][0]
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
                     "extras": extra_keys, "n_refs": len(ev.refs_std),
                     "multi": ev.multi, "secs": round(time.time() - t0, 1)}
        gen.save(out / f"{key}__gen.png")
        label_strip([("PROFILE INPUT", inp),
                     (f"GEN multi-photo cos={cosv} nme={nme}", gen),
                     ("TRUE FRONTAL", tgt)], height=460).save(out / f"{key}__board.png")
        print(f"[{key}] cos={cosv} nme={nme} refs={len(ev.refs_std)} "
              f"extras={extra_keys}", flush=True)

    est = round(renders * EST_PRICE["qwen"], 2)
    scored = [r for r in rows.values() if r.get("arcface_cos") is not None]
    coss = sorted(r["arcface_cos"] for r in scored)
    summary = {"tag": TAG, "seed": SEED, "n": len(scored), "renders": renders,
               "est_cost_usd": est,
               "arcface_cos_mean_subset": round(float(np.mean(coss)), 4) if coss else None}
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")
    print("\nSUMMARY " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
