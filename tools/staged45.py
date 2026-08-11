"""staged45 — the "strong version" staged-rotation pipeline (2026-08-06).

Stage 1: Qwen-2511 Multiple-Angles turns the profile crop 45 deg toward camera
         (both 45 and 315 are rendered; the more frontal result — smallest
         profile_ratio — is kept, so facing direction never has to be guessed).
Stage 2: the frontalizer gets TWO references: [original profile crop, best 45deg
         intermediate]. The real evidence stays in the reference set (identity
         anchor); the intermediate is only a pose hint. This is the dual-ref
         "strong version" — the 45-only "weak version" is rendered too, to
         prove the difference is real.

Arms per image (baselines already measured in Model_Compare_TestFile):
  B  qwen-2511  [profile, 45]   dual-ref            <- the idea under test
  C  qwen-2511  [45] only       weak version        <- expected worse
  D  flux-dev   [profile, 45]   dual-ref alternative
Direct single-ref baselines: qwen / flux from the Test File run.

Scores: buffalo_l cosine vs truth + frontality (profile_ratio) of every output,
plus the stage-1 intermediate scored on its own (how much identity survives).

Usage:  python tools/staged45.py [--mock] [--only KEY] [--seed N]
"""
from __future__ import annotations

import argparse
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
from compare_models_2511 import v8_report, v8_prompt, call_edit, QWEN, ANGLES  # noqa: E402
from generate_pure_profile import profile_ratio  # noqa: E402
from generate_testfile import build_images, extract_man_truths  # noqa: E402

TF = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_TestFile"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Staged45"

# one weak CCTV input + one clean profile: tests both halves of the prediction
KEYS = {
    "man-cctv2": {"subject": "man", "baseline_qwen": 0.4569, "baseline_flux": 0.3070},
    "pair4-c": {"subject": "pair4", "baseline_qwen": 0.5588, "baseline_flux": 0.4429},
}

POSE_NOTE = (
    " The second reference image is a rough synthetic preview of the head turned part-way "
    "toward the camera — use it only as a pose guide; every identity detail (face shape, "
    "features, skin, hair) must come from the first photograph, which is the only evidence."
)


def frontality(img: Image.Image, detector) -> float | None:
    faces = detector(img) or []
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return round(profile_ratio(best), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default=None, choices=sorted(KEYS))
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    images = build_images(extract_man_truths(detector))
    report: dict[str, dict] = {}

    def render_angles(url: str, angle: float, w: int, h: int) -> Image.Image:
        return call_edit(ANGLES, {
            "image_urls": [url], "seed": args.seed, "horizontal_angle": angle,
            "vertical_angle": 0, "zoom": 5, "image_size": {"width": w, "height": h},
            "num_images": 1, "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular"}, fal_provider.timeout())[0]

    def render_qwen(urls: list[str], prompt: str, w: int, h: int) -> Image.Image:
        return call_edit(QWEN, {
            "prompt": prompt, "image_urls": urls, "seed": args.seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular"}, fal_provider.timeout())[0]

    for key in (sorted(KEYS) if not args.only else [args.only]):
        spec = KEYS[key]
        ref = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        truth = Image.open(images[key]["truth"]).convert("RGB")
        prompt = v8_prompt(v8_report(spec["subject"]))
        w, h = fal_provider.fit_size(ref.width, ref.height)
        ref_url = fal_provider.to_data_url(ref)
        row: dict = {"baselines_direct_single_ref": {
            "qwen": spec["baseline_qwen"], "flux": spec["baseline_flux"]}}

        # ---------- stage 1: 45deg both directions, keep the most frontal ----------
        inters: list[tuple[float, Image.Image, float | None]] = []
        for angle in (45.0, 315.0):
            path = OUT / f"{key}__s1_angle{int(angle)}.png"
            try:
                img = (Image.open(path).convert("RGB") if path.exists()
                       else ref.resize((w, h)) if args.mock
                       else render_angles(ref_url, angle, w, h))
                img.save(path)
                fr = frontality(img, detector)
                inters.append((angle, img, fr))
                print(f"[{key}] s1 angle{int(angle)}: frontality_ratio={fr} "
                      f"vs_truth={vs_image(img, truth, embed)} vs_input={vs_image(img, ref, embed)}", flush=True)
            except Exception as exc:
                print(f"[{key}] s1 angle{int(angle)} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit("ABORT — fal balance exhausted")
        if not inters:
            continue
        # most frontal = smallest profile_ratio (None sorts last)
        angle, inter, fr = min(inters, key=lambda t: (t[2] is None, t[2]))
        inter_url = fal_provider.to_data_url(inter)
        row["stage1"] = {"chosen_angle": angle, "frontality_ratio": fr,
                         "vs_truth": vs_image(inter, truth, embed),
                         "vs_input": vs_image(inter, ref, embed)}
        print(f"[{key}] stage1 chose angle {int(angle)}", flush=True)

        # ---------- stage 2 arms ----------
        arms = {
            "B_qwen_dualref": lambda: render_qwen([ref_url, inter_url], prompt + POSE_NOTE, w, h),
            "C_qwen_45only": lambda: render_qwen([inter_url], prompt, w, h),
            "D_flux_dualref": lambda: fal_provider.edit_images(
                prompt + POSE_NOTE, [ref, inter], width=w, height=h, seed=args.seed)[0][0],
        }
        strip = [("PROFILE (evidence)", ref), (f"45deg intermediate (angle {int(angle)})", inter)]
        for tag, run in arms.items():
            path = OUT / f"{key}__{tag}.png"
            try:
                img = (Image.open(path).convert("RGB") if path.exists()
                       else ref.resize((w, h)) if args.mock else run())
                img.save(path)
                score = vs_image(img, truth, embed)
                row[tag] = {"vs_truth": score, "vs_input": vs_image(img, ref, embed),
                            "frontality_ratio": frontality(img, detector)}
                strip.append((f"{tag}  vs_truth={score}", img))
                print(f"[{key}] {tag}: vs_truth={score} (direct qwen baseline {spec['baseline_qwen']})", flush=True)
            except Exception as exc:
                print(f"[{key}] {tag} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit("ABORT — fal balance exhausted")
        strip.append(("REAL TRUTH (never sent)", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD.png")
        report[key] = row

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
