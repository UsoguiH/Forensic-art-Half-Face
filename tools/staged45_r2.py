"""staged45_r2 — round 2 of the staged-rotation experiment.

Round-1 lessons applied:
  B2  qwen dual-ref  [profile, 45-intermediate picked by IDENTITY not frontality]
  B3  qwen triple-ref [profile, mirror(profile), 45-hi]  (mirror = measured helper)
  E   direct qwen single-ref, seeds 202 & 303 -> with round-1's seed-101 direct
      baseline this gives best-of-3 at the SAME 3-render budget staged spends.

Verdict rule: staged is only "the best version" if B2/B3 beat best-of-3 direct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import fal_provider  # noqa: E402
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from compare_models_2511 import v8_report, v8_prompt, call_edit, QWEN  # noqa: E402
from generate_testfile import build_images, extract_man_truths  # noqa: E402
from staged45 import KEYS, POSE_NOTE, frontality, OUT  # noqa: E402

TF = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_TestFile"


def main() -> None:
    detector, embed = load_engine(False)
    images = build_images(extract_man_truths(detector))
    r1 = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    report: dict[str, dict] = {}

    def qwen(urls, prompt, w, h, seed):
        return call_edit(QWEN, {
            "prompt": prompt, "image_urls": urls, "seed": seed,
            "image_size": {"width": w, "height": h}, "num_images": 1,
            "output_format": "png", "sync_mode": True,
            "enable_safety_checker": False, "acceleration": "regular"}, fal_provider.timeout())[0]

    for key, spec in KEYS.items():
        ref = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        truth = Image.open(images[key]["truth"]).convert("RGB")
        prompt = v8_prompt(v8_report(spec["subject"]))
        w, h = fal_provider.fit_size(ref.width, ref.height)
        ref_url = fal_provider.to_data_url(ref)

        # pick the 45-intermediate by IDENTITY (vs_input), not frontality
        cands = []
        for angle in (45, 315):
            p = OUT / f"{key}__s1_angle{angle}.png"
            if p.exists():
                img = Image.open(p).convert("RGB")
                cands.append((vs_image(img, ref, embed) or -1, angle, img))
        cands.sort(reverse=True, key=lambda t: t[0])
        vi, angle, inter = cands[0]
        inter_url = fal_provider.to_data_url(inter)
        print(f"[{key}] identity-picked intermediate: angle{angle} (vs_input={vi})", flush=True)

        mirror_url = fal_provider.to_data_url(ImageOps.mirror(ref))
        row: dict = {"intermediate": {"angle": angle, "vs_input": vi}}
        strip = [("PROFILE", ref), (f"45-hi (angle {angle})", inter)]

        jobs = {
            "B2_dualref_hi": lambda s: qwen([ref_url, inter_url], prompt + POSE_NOTE, w, h, s),
            "B3_tripleref": lambda s: qwen([ref_url, mirror_url, inter_url], prompt + POSE_NOTE, w, h, s),
            "E_direct_202": lambda s: qwen([ref_url], prompt, w, h, 202),
            "E_direct_303": lambda s: qwen([ref_url], prompt, w, h, 303),
        }
        for tag, run in jobs.items():
            path = OUT / f"{key}__{tag}.png"
            try:
                img = Image.open(path).convert("RGB") if path.exists() else run(101)
                img.save(path)
                score = vs_image(img, truth, embed)
                row[tag] = {"vs_truth": score, "vs_input": vs_image(img, ref, embed),
                            "frontality_ratio": frontality(img, detector)}
                strip.append((f"{tag}  vs_truth={score}", img))
                print(f"[{key}] {tag}: vs_truth={score}", flush=True)
            except Exception as exc:
                print(f"[{key}] {tag} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit("ABORT — fal balance exhausted")

        direct101 = spec["baseline_qwen"]
        directs = [direct101] + [row[t]["vs_truth"] for t in ("E_direct_202", "E_direct_303")
                                 if t in row and row[t]["vs_truth"] is not None]
        staged = [v["vs_truth"] for t, v in row.items()
                  if t.startswith("B") and isinstance(v, dict) and v.get("vs_truth") is not None]
        r1_staged = [v.get("vs_truth") for t, v in r1.get(key, {}).items()
                     if t.startswith(("B", "C", "D")) and isinstance(v, dict)]
        row["verdict"] = {
            "direct_best_of_3": max(directs), "staged_best_all_rounds": max(staged + [s for s in r1_staged if s]),
            "staged_wins": max(staged + [s for s in r1_staged if s]) > max(directs),
        }
        strip.append(("REAL TRUTH", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD_r2.png")
        report[key] = row
        print(f"[{key}] VERDICT: {row['verdict']}", flush=True)

    (OUT / "report_r2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["verdict"] for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
