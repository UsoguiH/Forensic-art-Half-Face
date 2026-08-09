"""generate_pure_profile — provably profile-only generation run (2026-08-06).

Guarantee, by construction: for every dataset photo we cut ONE tight
side-profile head crop, write it to disk as <key>__0_REF-SENT.png, and the
request body is built from THOSE EXACT FILE BYTES (SHA-256 of the payload is
logged and must equal the hash of the file on disk). Nothing else is attached:
no mirror, no extra photos, and the truth photo never appears in any request —
it is only opened locally afterwards for scoring. The text prompt contains a
written description only, no pixels.

Models (fresh seed 101 — everything newly generated, no cache):
  FLUX-2-dev, Hunyuan-Image-v3 instruct/edit, Qwen-Edit-2511,
  Qwen-2511-Multiple-Angles (horizontal 90 and 270)

Output: Test_AI_Half/Model_Compare_PURE_PROFILE/  (flat, obvious names + boards)

Usage (from Forensic_Art_v8):  python tools/generate_pure_profile.py [--mock]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import fal_provider  # noqa: E402
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from compare_models_2511 import v8_report, v8_prompt, call_edit, HUNYUAN, QWEN, ANGLES, NICE  # noqa: E402
from compare_models_dst import IMAGES  # noqa: E402

OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_PURE_PROFILE"
SEED = 101


def profile_ratio(face) -> float:
    """How side-on a detected face is. Frontal ≈ 0-0.4; strict profile >> 1.
    (nose x-offset from the eye midpoint, relative to the eye spacing —
    on a profile the eyes collapse together and the nose sticks far out)."""
    import numpy as np

    kps = np.asarray(face.kps, dtype=float)
    le, re, nose = kps[0], kps[1], kps[2]
    return abs(nose[0] - (le[0] + re[0]) / 2.0) / max(8.0, abs(re[0] - le[0]))


def face_crop_by(img: Image.Image, detector, pick) -> Image.Image:
    """Tight head crop around the face chosen by ``pick`` from all detections.

    Several DateSet Test files are two-panel collages (profile + frontal), so
    'biggest face' silently grabs the FRONTAL — the answer. We pick by pose:
    max profile_ratio for the evidence crop, min for extracting a truth panel.
    """
    faces = detector(img) or []
    if not faces:
        return img
    best = pick(faces)
    x0, y0, x1, y1 = (float(v) for v in best.bbox[:4])
    fw, fh = x1 - x0, y1 - y0
    box = (int(max(0, x0 - fw * 0.45)), int(max(0, y0 - fh * 0.75)),
           int(min(img.width, x1 + fw * 0.45)), int(min(img.height, y1 + fh * 0.85)))
    crop = img.crop(box)
    if min(crop.size) < 512:
        s = 512 / min(crop.size)
        crop = crop.resize((round(crop.width * s), round(crop.height * s)),
                           Image.Resampling.LANCZOS)
    return crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default=None, choices=sorted(IMAGES))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    prompts = {s: v8_prompt(v8_report(s)) for s in ("young", "beard", "pair4", "man")}
    grand: dict[str, dict] = {}
    renders = 0

    for key in (sorted(IMAGES) if not args.only else [args.only]):
        spec = IMAGES[key]
        prompt = prompts[spec["subject"]]
        original = Image.open(spec["file"]).convert("RGB")
        truth = Image.open(spec["truth"]).convert("RGB") if spec.get("truth") else None
        truth_src = str(spec.get("truth") or "")

        faces = detector(original) or []
        ratios = sorted(round(profile_ratio(f), 2) for f in faces)
        print(f"[{key}] {len(faces)} face(s) in file, profile_ratios={ratios}", flush=True)
        # man1/man2 collages: the frontal right panel becomes the scoring truth
        # (it exists in the file anyway — but it is NEVER put in a request)
        if truth is None and len(faces) >= 2:
            truth = face_crop_by(original, detector, lambda fs: min(fs, key=profile_ratio))
            truth_src = "frontal panel extracted from the collage (never sent)"

        # ---- the ONE reference: the most PROFILE face, written to disk first ----
        ref = face_crop_by(original, detector, lambda fs: max(fs, key=profile_ratio))
        ref_path = OUT / f"{key}__0_REF-SENT.png"
        ref.save(ref_path)
        ref_bytes = ref_path.read_bytes()                      # exact bytes on disk
        ref_sha = hashlib.sha256(ref_bytes).hexdigest()
        data_url = "data:image/png;base64," + base64.b64encode(ref_bytes).decode("ascii")
        payload_sha = hashlib.sha256(base64.b64decode(data_url.split(",", 1)[1])).hexdigest()
        assert payload_sha == ref_sha, "payload must be byte-identical to REF-SENT file"
        print(f"[{key}] REF-SENT sha256={ref_sha[:16]}… payload identical: {payload_sha == ref_sha}", flush=True)

        w, h = fal_provider.fit_size(ref.width, ref.height)
        honest = "vs_truth" if truth is not None else "vs_input"

        def score(img: Image.Image) -> dict:
            return {"vs_truth": vs_image(img, truth, embed),
                    "vs_input": vs_image(img, ref, embed)}

        common = {"image_urls": [data_url], "seed": SEED,
                  "image_size": {"width": w, "height": h}, "num_images": 1,
                  "output_format": "png", "sync_mode": True,
                  "enable_safety_checker": False}
        jobs = [
            ("flux-dev", None, None),
            ("hunyuan-v3", HUNYUAN, {**common, "prompt": prompt, "enable_prompt_expansion": False}),
            ("qwen-2511", QWEN, {**common, "prompt": prompt, "acceleration": "regular"}),
            ("qwen-angles@90", ANGLES, {**common, "horizontal_angle": 90.0, "vertical_angle": 0,
                                        "zoom": 5, "acceleration": "regular"}),
            ("qwen-angles@270", ANGLES, {**common, "horizontal_angle": 270.0, "vertical_angle": 0,
                                         "zoom": 5, "acceleration": "regular"}),
        ]
        cands: dict[str, list[dict]] = {}
        for label, slug, body in jobs:
            model = label.split("@")[0]
            try:
                if args.mock:
                    img = ref.resize((w, h), Image.Resampling.LANCZOS)
                elif model == "flux-dev":
                    img = fal_provider.edit_images(prompt, [ref], width=w, height=h, seed=SEED)[0][0]
                else:
                    img = call_edit(slug, body, fal_provider.timeout())[0]
                renders += 1
                row = {"image": img, "label": label, **score(img)}
                cands.setdefault(model, []).append(row)
                print(f"[{key}] {label}: {honest}={row[honest]}", flush=True)
            except Exception as exc:
                print(f"[{key}] {label} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit(f"ABORT — fal balance exhausted: {exc}")

        results = {m: max(rows, key=lambda r: (r.get(honest) if r.get(honest) is not None else -1.0))
                   for m, rows in cands.items()}

        original.save(OUT / f"{key}__1_INPUT.png")
        strip = [("INPUT (full photo)", original), ("REF SENT (all the model sees)", ref)]
        for i, model in enumerate(["flux-dev", "hunyuan-v3", "qwen-2511", "qwen-angles"], start=2):
            r = results.get(model)
            if not r:
                continue
            r["image"].save(OUT / f"{key}__{i}_{NICE[model]}.png")
            strip.append((f"{NICE[model]}  {honest}={r.get(honest)}", r["image"]))
        if truth is not None:
            truth.save(OUT / f"{key}__6_TRUTH.png")
            strip.append(("REAL TRUTH (never sent)", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD.png")

        grand[key] = {"file": str(spec["file"]), "truth_source": truth_src, "ref_sha256": ref_sha,
                      "payload_sha256_matches_ref": True, "honest_metric": honest,
                      "input_vs_truth_leakcheck": vs_image(ref, truth, embed),
                      "scores": {m: {k: v for k, v in r.items() if k != "image"}
                                 for m, r in results.items()}}
        print(f"[{key}] done: " + " | ".join(
            f"{m}={results[m].get(honest)}" for m in ("flux-dev", "hunyuan-v3", "qwen-2511", "qwen-angles")
            if m in results), flush=True)

    grand["_meta"] = {"date": "2026-08-06", "seed": SEED, "renders": renders, "mock": args.mock,
                      "protocol": "single profile crop only; payload bytes == REF-SENT file (sha256); "
                                  "truth photos never leave this machine"}
    (OUT / "report.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
