"""compare_models_2511 — DateSet Test model shoot-out (2026-08-06).

Compares three fal.ai editing models against the FLUX.2 [dev] v8 champions on
the ---DateSet Test--- benchmark, same evidence refs + same v8 prompts:

  hunyuan-v3    fal-ai/hunyuan-image/v3/instruct/edit
  qwen-2511     fal-ai/qwen-image-edit-2511
  qwen-angles   fal-ai/qwen-image-edit-2511-multiple-angles  (camera params, no prompt)

FLUX dev is NOT re-rendered — the v8-base winners (already paid for) are the
baseline. Every candidate is scored with the same local buffalo_l metric used
by v8 (vs REAL truth where one exists, vs fused evidence anchor for 'man').

Output: one flat folder Test_AI_Half/Model_Compare_2511/ with obvious names
(<subject>__2_FLUX-2-dev.png, <subject>__3_Hunyuan-Image-v3.png, ...), a
labeled board strip per subject, raw/ candidates, README.md + report.json.

Usage (from Forensic_Art_v8):  python tools/compare_models_2511.py [--mock]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import fal_provider  # noqa: E402
from halfface import build_evidence  # noqa: E402
from halfface.rank import vs_image, unit, cos  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from iterate_v8 import SUBJECTS  # noqa: E402

V8_ROOT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "v8"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_2511"

HUNYUAN = "fal-ai/hunyuan-image/v3/instruct/edit"
QWEN = "fal-ai/qwen-image-edit-2511"
ANGLES = "fal-ai/qwen-image-edit-2511-multiple-angles"

SEEDS = (7, 21)

# Display names — these become the file names the user compares by.
NICE = {
    "flux-dev": "FLUX-2-dev",
    "hunyuan-v3": "Hunyuan-Image-v3",
    "qwen-2511": "Qwen-Edit-2511",
    "qwen-angles": "Qwen-Multiple-Angles",
}


def v8_report(subject: str) -> dict:
    return json.loads((V8_ROOT / f"{subject}__v8-base" / "report.json").read_text(encoding="utf-8"))


def v8_prompt(report: dict) -> str:
    """The prompt of the style the v8 pick actually used (production choice)."""
    style = report.get("picked", {}).get("style") or "forensic"
    prompts = report.get("prompts", {})
    return prompts.get(f"{style}/std") or prompts.get("forensic/std") or next(iter(prompts.values()))


def call_edit(slug: str, body: dict, timeout: float) -> list[Image.Image]:
    payload = fal_provider._post(slug, body, timeout)
    return fal_provider._decode(payload, timeout)


def render(model: str, prompt: str, refs: list[Image.Image], w: int, h: int,
           seed: int, mock: bool, angle: float | None = None) -> Image.Image:
    if mock:
        return refs[0].convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    urls = [fal_provider.to_data_url(r) for r in refs]
    t = fal_provider.timeout()
    if model == "hunyuan-v3":
        body = {
            "prompt": prompt, "image_urls": urls, "seed": seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_prompt_expansion": False, "enable_safety_checker": False,
        }
        return call_edit(HUNYUAN, body, t)[0]
    if model == "qwen-2511":
        body = {
            "prompt": prompt, "image_urls": urls, "seed": seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular",
        }
        return call_edit(QWEN, body, t)[0]
    if model == "qwen-angles":
        body = {
            "image_urls": urls[:1], "seed": seed,
            "horizontal_angle": float(angle or 0), "vertical_angle": 0, "zoom": 5,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular",
        }
        return call_edit(ANGLES, body, t)[0]
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--subject", default="all", choices=sorted(SUBJECTS) + ["all"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    detector, embed = load_engine(False)
    grand: dict[str, dict] = {}
    requests_made = 0

    for name in (sorted(SUBJECTS) if args.subject == "all" else [args.subject]):
        spec = SUBJECTS[name]
        report = v8_report(name)
        prompt = v8_prompt(report)
        primary = Image.open(spec["primary"]).convert("RGB")
        extras = [Image.open(p).convert("RGB") for p in spec["extras"]]
        truth = Image.open(spec["truth"]).convert("RGB") if spec.get("truth") else None
        target = Image.open(spec["target"]).convert("RGB") if spec.get("target") else None

        ev = build_evidence(primary, extras, detector=detector, embed=embed, who="a man")
        w, h = fal_provider.fit_size(ev.working.width, ev.working.height)
        fused = getattr(ev, "fused", None)

        def score(img: Image.Image) -> dict:
            row = {
                "vs_truth": vs_image(img, truth, embed),
                "vs_target": vs_image(img, target, embed),
                "sim_fused": None,
            }
            if fused is not None:
                e = unit(embed(img))
                if e is not None:
                    row["sim_fused"] = round(float(cos(e, unit(fused))), 4)
            return row

        honest = "vs_truth" if truth is not None else "sim_fused"
        results: dict[str, dict] = {}

        # --- baseline: existing FLUX.2 dev v8 winner (no new spend) ---
        flux_img = Image.open(V8_ROOT / f"{name}__v8-base" / "winner.png").convert("RGB")
        results["flux-dev"] = {"image": flux_img, **score(flux_img),
                               "note": "v8-base picked winner (pre-existing)"}

        # --- the three challengers ---
        plans: list[tuple[str, dict]] = []
        for seed in SEEDS:
            plans.append(("hunyuan-v3", {"seed": seed, "refs": ev.refs_std}))
            plans.append(("qwen-2511", {"seed": seed, "refs": ev.refs_std}))
        for angle in (90.0, -90.0):
            plans.append(("qwen-angles", {"seed": SEEDS[0], "refs": [ev.working], "angle": angle}))

        cands: dict[str, list[dict]] = {}
        for model, p in plans:
            tag = f"{name}__{NICE[model]}__seed{p['seed']}" + (
                f"_angle{int(p['angle'])}" if "angle" in p else "")
            try:
                t0 = time.time()
                img = render(model, prompt, p["refs"], w, h, p["seed"], args.mock, p.get("angle"))
                requests_made += 1
                row = {"image": img, "seed": p["seed"], "angle": p.get("angle"),
                       "seconds": round(time.time() - t0, 1), **score(img)}
                img.save(OUT / "raw" / f"{tag}.png")
                cands.setdefault(model, []).append(row)
                print(f"[{name}] {tag}: {honest}={row[honest]} ({row['seconds']}s)", flush=True)
            except Exception as exc:
                print(f"[{name}] {tag} FAILED: {exc}", flush=True)
                text = str(exc).lower()
                if "balance" in text or "locked" in text or "402" in text:
                    raise SystemExit(f"ABORT — fal balance exhausted: {exc}")

        for model, rows in cands.items():
            key = lambda r: (r.get(honest) if r.get(honest) is not None else -1.0)
            results[model] = max(rows, key=key)

        # --- flat, obviously-named comparison files ---
        order = ["flux-dev", "hunyuan-v3", "qwen-2511", "qwen-angles"]
        primary.save(OUT / f"{name}__1_INPUT.png")
        strip = [("INPUT", primary)]
        for i, model in enumerate(order, start=2):
            r = results.get(model)
            if not r:
                continue
            fname = f"{name}__{i}_{NICE[model]}.png"
            r["image"].save(OUT / fname)
            label = f"{NICE[model]}  {honest}={r.get(honest)}"
            strip.append((label, r["image"]))
        if truth is not None:
            truth.save(OUT / f"{name}__6_TRUTH.png")
            strip.append(("REAL TRUTH", truth))
        elif target is not None:
            strip.append(("AI target (legacy)", target))
        label_strip(strip, height=560).save(OUT / f"{name}__BOARD.png")

        grand[name] = {
            "honest_metric": honest,
            "prompt_style": report.get("picked", {}).get("style"),
            "scores": {m: {k: v for k, v in r.items() if k != "image"}
                       for m, r in results.items()},
        }
        print(f"[{name}] done: " + " | ".join(
            f"{m}={results[m].get(honest)}" for m in order if m in results), flush=True)

    grand["_meta"] = {"date": "2026-08-06", "requests_made": requests_made,
                      "seeds": SEEDS, "endpoints": {"hunyuan-v3": HUNYUAN,
                      "qwen-2511": QWEN, "qwen-angles": ANGLES},
                      "mock": args.mock}
    (OUT / "report.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("scores") and {m: s.get(v["honest_metric"]) for m, s in v["scores"].items()}
                      for k, v in grand.items() if not k.startswith("_")}, indent=1))


if __name__ == "__main__":
    main()
