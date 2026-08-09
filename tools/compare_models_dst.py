"""compare_models_dst — per-image model shoot-out on ---DateSet Test--- (2026-08-06).

7 photos from the DateSet Test folder, each run through 4 models with the SAME
single-image evidence (refs_std = [working crop, mirror]) and the subject's v8
prompt:

  FLUX-2-dev            fal-ai/flux-2/edit                       (1 seed)
  Hunyuan-Image-v3      fal-ai/hunyuan-image/v3/instruct/edit    (1 seed)
  Qwen-Edit-2511        fal-ai/qwen-image-edit-2511              (1 seed)
  Qwen-Multiple-Angles  fal-ai/qwen-image-edit-2511-multiple-angles (+90 / -90)

Frugality: any render already sitting in raw/ (from the aborted subject-level
run) is reused instead of re-paid, and extra seeds found there join the
best-of pool. beard/pair4 FLUX baselines are the v8-base winners — v8 built
them from the identical single-primary evidence, so they are the same
experiment already paid for.

Scores: local buffalo_l cosine vs REAL truth (man cluster has none — vs_input
and the legacy AI target are reported instead, clearly marked report-only).

Usage (from Forensic_Art_v8):  python tools/compare_models_dst.py [--mock]
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
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from compare_models_2511 import v8_report, v8_prompt, call_edit, HUNYUAN, QWEN, ANGLES, NICE  # noqa: E402

HALF = Path(r"D:\Claude\Forensic_ART_Original\Forensic_Art\half photo")
DST = HALF / "---DateSet Test---"
SCOPE = HALF / "Scope"
V8_ROOT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "v8"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_2511"

SEED = 7

IMAGES: dict[str, dict] = {
    "man1":   {"file": DST / "man 1.png", "subject": "man",
               "truth": None, "target": SCOPE / "tgt_man2.png"},
    "man2":   {"file": DST / "man 2.png", "subject": "man",
               "truth": None, "target": SCOPE / "tgt_man2.png"},
    "young6": {"file": DST / "young 6.png", "subject": "young",
               "truth": DST / "original 6.png", "target": None},
    "young7": {"file": DST / "young 7.png", "subject": "young",
               "truth": DST / "original 6.png", "target": None},
    "cctv":   {"file": DST / "Screenshot 2026-08-04 093224.png", "subject": "young",
               "truth": DST / "original 6.png", "target": None},
    "pair4":  {"file": DST / "WhatsApp Image 2026-08-02 at 6.38.19 AM.jpeg", "subject": "pair4",
               "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg", "target": None,
               "flux_v8_winner": V8_ROOT / "pair4__v8-base" / "winner.png"},
    "beard":  {"file": DST / "WhatsApp Image 2026-08-02 at 9.44.08 AM.jpeg", "subject": "beard",
               "truth": HALF / "WhatsApp Image 2026-08-02 at 9.44.07 AM.jpeg", "target": None,
               "flux_v8_winner": V8_ROOT / "beard__v8-base" / "winner.png"},
}


def render_one(model: str, prompt: str, refs: list[Image.Image], w: int, h: int,
               seed: int, angle: float | None, mock: bool) -> Image.Image:
    if mock:
        return refs[0].convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    t = fal_provider.timeout()
    urls = [fal_provider.to_data_url(r) for r in refs]
    if model == "flux-dev":
        images, _slug = fal_provider.edit_images(prompt, refs, width=w, height=h, seed=seed)
        return images[0]
    if model == "hunyuan-v3":
        return call_edit(HUNYUAN, {
            "prompt": prompt, "image_urls": urls, "seed": seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_prompt_expansion": False, "enable_safety_checker": False}, t)[0]
    if model == "qwen-2511":
        return call_edit(QWEN, {
            "prompt": prompt, "image_urls": urls, "seed": seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular"}, t)[0]
    if model == "qwen-angles":
        return call_edit(ANGLES, {
            "image_urls": urls[:1], "seed": seed,
            "horizontal_angle": float(angle or 0), "vertical_angle": 0, "zoom": 5,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular"}, t)[0]
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default=None, choices=sorted(IMAGES))
    ap.add_argument("--strict", action="store_true",
                    help="purity run: each model gets ONLY the single tight profile "
                         "crop (no mirror, no extras); the exact ref sent is saved "
                         "as proof; flux-dev + qwen-2511 only")
    args = ap.parse_args()

    global OUT
    if args.strict:
        OUT = OUT / "strict_profile"
    OUT.mkdir(parents=True, exist_ok=True)
    # mock renders must not land in the reusable raw/ cache
    raw = OUT / ("raw_mock" if args.mock else "raw")
    raw.mkdir(exist_ok=True)
    # subject-level leftovers from the aborted first run that no longer have a
    # matching per-image entry ("man" multi-ref renders) stay in raw/ untouched;
    # stale named/board files are re-written below.
    detector, embed = load_engine(False)

    grand: dict[str, dict] = {}
    new_renders = 0
    prompts = {s: v8_prompt(v8_report(s)) for s in ("young", "beard", "pair4", "man")}

    for key in (sorted(IMAGES) if not args.only else [args.only]):
        spec = IMAGES[key]
        prompt = prompts[spec["subject"]]
        primary = Image.open(spec["file"]).convert("RGB")
        truth = Image.open(spec["truth"]).convert("RGB") if spec.get("truth") else None
        target = Image.open(spec["target"]).convert("RGB") if spec.get("target") else None

        ev = build_evidence(primary, [], detector=detector, embed=embed, who="a man")
        w, h = fal_provider.fit_size(ev.working.width, ev.working.height)
        honest = "vs_truth" if truth is not None else "vs_input"

        def score(img: Image.Image) -> dict:
            return {"vs_truth": vs_image(img, truth, embed),
                    "vs_input": vs_image(img, primary, embed),
                    "vs_target_legacy": vs_image(img, target, embed)}

        # (model, seed, angle, refs)
        if args.strict:
            # purity protocol: ONE reference only — the tight profile crop.
            # v8 winners came from [working, mirror] refs, so flux re-renders too.
            plans = [
                ("flux-dev", SEED, None, [ev.working]),
                ("qwen-2511", SEED, None, [ev.working]),
            ]
            spec = {**spec, "flux_v8_winner": None}
            ev.working.save(OUT / f"{key}__0_REF-SENT.png")
        else:
            plans = [
                ("flux-dev", SEED, None, ev.refs_std),
                ("hunyuan-v3", SEED, None, ev.refs_std),
                ("qwen-2511", SEED, None, ev.refs_std),
                # endpoint requires horizontal_angle in [0, 360): 90 = one way,
                # 270 = the other, so both profile directions get a frontal try
                ("qwen-angles", SEED, 90.0, [ev.working]),
                ("qwen-angles", SEED, 270.0, [ev.working]),
            ]
        cands: dict[str, list[dict]] = {}
        for model, seed, angle, refs in plans:
            tag = f"{key}__{NICE[model]}__seed{seed}" + (f"_angle{int(angle)}" if angle else "")
            path = raw / f"{tag}.png"
            # v8 FLUX winners for beard/pair4 came from this exact evidence — reuse.
            if model == "flux-dev" and spec.get("flux_v8_winner") and not path.exists():
                Image.open(spec["flux_v8_winner"]).convert("RGB").save(path)
            try:
                if path.exists():
                    img, reused = Image.open(path).convert("RGB"), True
                else:
                    t0 = time.time()
                    img = render_one(model, prompt, refs, w, h, seed, angle, args.mock)
                    new_renders += 1
                    img.save(path)
                    reused = False
                row = {"image": img, "seed": seed, "angle": angle, "reused": reused, **score(img)}
                cands.setdefault(model, []).append(row)
                print(f"[{key}] {tag}{' (reused)' if reused else ''}: {honest}={row[honest]}", flush=True)
            except Exception as exc:
                print(f"[{key}] {tag} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit(f"ABORT — fal balance exhausted: {exc}")

        # free extra seeds left in raw/ by the first run join the best-of pool
        planned = {f"{key}__{NICE[m]}__seed{s}" + (f"_angle{int(a)}" if a else "") + ".png"
                   for m, s, a, _ in plans}
        for extra in raw.glob(f"{key}__*__seed*.png"):
            model = next((m for m, n in NICE.items() if f"__{n}__" in extra.name), None)
            if model and extra.name not in planned:
                img = Image.open(extra).convert("RGB")
                row = {"image": img, "seed": None, "angle": None, "reused": True,
                       "raw_file": extra.name, **score(img)}
                cands.setdefault(model, []).append(row)
                print(f"[{key}] pooled extra {extra.name}: {honest}={row[honest]}", flush=True)

        results = {m: max(rows, key=lambda r: (r.get(honest) if r.get(honest) is not None else -1.0))
                   for m, rows in cands.items()}

        order = ["flux-dev", "hunyuan-v3", "qwen-2511", "qwen-angles"]
        primary.save(OUT / f"{key}__1_INPUT.png")
        strip = [("INPUT", primary)]
        for i, model in enumerate(order, start=2):
            r = results.get(model)
            if not r:
                continue
            r["image"].save(OUT / f"{key}__{i}_{NICE[model]}.png")
            strip.append((f"{NICE[model]}  {honest}={r.get(honest)}", r["image"]))
        if truth is not None:
            truth.save(OUT / f"{key}__6_TRUTH.png")
            strip.append(("REAL TRUTH", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD.png")

        # leak check: if the input were derived from the truth photo this would be ~0.99
        input_vs_truth = vs_image(primary, truth, embed)
        print(f"[{key}] leak-check input_vs_truth={input_vs_truth}", flush=True)
        grand[key] = {"file": str(spec["file"]), "subject_prompt": spec["subject"],
                      "honest_metric": honest, "input_vs_truth": input_vs_truth,
                      "scores": {m: {k: v for k, v in r.items() if k != "image"}
                                 for m, r in results.items()}}
        print(f"[{key}] done: " + " | ".join(f"{m}={results[m].get(honest)}"
              for m in order if m in results), flush=True)

    grand["_meta"] = {"date": "2026-08-06", "new_renders": new_renders, "seed": SEED,
                      "endpoints": {"flux-dev": "fal-ai/flux-2/edit", "hunyuan-v3": HUNYUAN,
                                    "qwen-2511": QWEN, "qwen-angles": ANGLES},
                      "mock": args.mock,
                      "note": "man cluster has no real frontal truth; vs_input/vs_target_legacy are report-only"}
    (OUT / "report.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: s.get(v["honest_metric"]) for m, s in v["scores"].items()}
                      for k, v in grand.items() if not k.startswith("_")}, indent=1))


if __name__ == "__main__":
    main()
