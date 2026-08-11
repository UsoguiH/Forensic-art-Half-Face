"""pick_newdataset — offline honest best-of-N picker over existing bench runs.

Combines the saved renders of several bench_newdataset runs (no new fal cost),
scores each candidate with the PRODUCTION ranker (frontality gate >= 0.55,
single-photo rank = min(sim-to-profile, sim-to-mirror) — truth never touched),
and reports picked-vs-truth + oracle-vs-truth per sample.

Usage (from Forensic_Art_v8):
  python tools/pick_newdataset.py --runs ITER1-qwen-s42 ITER2a-qwen-s202 --tag ITER2-bestof2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

from halfface import build_evidence  # noqa: E402
from halfface import rank  # noqa: E402
from halfface.rank import Candidate, vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from bench_newdataset import BENCH, SPLIT, load_pairs, landmark_nme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    out = BENCH / args.tag
    out.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)

    rows: dict[str, dict] = {}
    for key, inp_path, tgt_path in load_pairs():
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        ev = build_evidence(inp, None, detector=detector, embed=embed)

        cands = []
        for run in args.runs:
            p = BENCH / run / f"{key}__gen.png"
            if not p.exists():
                print(f"[{key}] MISSING candidate {p}", flush=True)
                continue
            c = Candidate(image=Image.open(p).convert("RGB"), style="forensic", seed=0)
            c.render_meta["run"] = run
            cands.append(rank.score(c, ev, detector, embed))
        if not cands:
            rows[key] = {"error": "no candidates"}
            continue

        picked = rank.pick(cands, ev)
        per = {c.render_meta["run"]: {"vs_truth": vs_image(c.image, tgt, embed),
                                      "rank_key": round(rank.rank_key(c, ev), 4),
                                      "frontality": round(c.frontality, 3),
                                      "gated": c.gated, "gate_reason": c.gate_reason}
               for c in cands}
        oracle_run, oracle = max(((r, d["vs_truth"]) for r, d in per.items()
                                  if d["vs_truth"] is not None),
                                 key=lambda t: t[1], default=(None, None))
        pcos = vs_image(picked.image, tgt, embed)
        pnme = landmark_nme(picked.image, tgt, detector)
        rows[key] = {"arcface_cos": pcos, "landmark_nme": pnme,
                     "picked_run": picked.render_meta["run"],
                     "oracle_run": oracle_run, "oracle_cos": oracle,
                     "picked_is_oracle": picked.render_meta["run"] == oracle_run,
                     "candidates": per}
        picked.image.save(out / f"{key}__gen.png")
        label_strip([("PROFILE INPUT", inp),
                     (f"PICKED {picked.render_meta['run']} cos={pcos} nme={pnme}", picked.image),
                     ("TRUE FRONTAL", tgt)], height=460).save(out / f"{key}__board.png")
        print(f"[{key}] picked={picked.render_meta['run']} cos={pcos} "
              f"oracle={oracle} ({oracle_run}) nme={pnme}", flush=True)

    scored = [r for r in rows.values() if r.get("arcface_cos") is not None]
    coss = sorted(r["arcface_cos"] for r in scored)
    nmes = [r["landmark_nme"] for r in scored if r.get("landmark_nme") is not None]
    oracles = [r["oracle_cos"] for r in scored if r.get("oracle_cos") is not None]
    summary = {"tag": args.tag, "runs": args.runs, "note": args.note, "n": len(scored),
               "arcface_cos": {"mean": round(float(np.mean(coss)), 4),
                               "median": round(float(np.median(coss)), 4),
                               "worst": round(float(coss[0]), 4)},
               "oracle_mean": round(float(np.mean(oracles)), 4),
               "picked_is_oracle": sum(1 for r in scored if r.get("picked_is_oracle")),
               "landmark_nme_mean": round(float(np.mean(nmes)), 4) if nmes else None,
               "below_0.4": [k for k, r in rows.items()
                             if (r.get("arcface_cos") or 1) < 0.4]}
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")

    with BENCH.joinpath("results_table.md").open("a", encoding="utf-8") as f:
        ac = summary["arcface_cos"]
        f.write(f"| {args.tag} | {args.note} | {summary['n']} | {ac['mean']} | "
                f"{ac['median']} | {ac['worst']} | {summary['landmark_nme_mean']} | 0 | 0.0 |\n")
    print("\nSUMMARY " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
