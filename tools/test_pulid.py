"""test_pulid — the embedding-injection route, tested honestly (2026-08-06).

fal-ai/flux-pulid = the only fal-hosted, ID-loss-trained embedding injector.
Research-recommended settings: id_weight=1.0 (fal cap), true_cfg=1,
start_step=4 (photoreal baseline) and start_step=0 (max fidelity variant).
Same strict protocol: the sha-verified profile crop is the only image sent;
scored vs the real truth (never sent). Baselines: qwen-2511 direct best
(beard 0.5359 / pair4-c 0.5958 best-of-3).
"""
from __future__ import annotations

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
from compare_models_2511 import v8_report, v8_prompt, call_edit  # noqa: E402
from generate_testfile import build_images, extract_man_truths  # noqa: E402

TF = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_TestFile"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Embedding_Research"
PULID = "fal-ai/flux-pulid"

SUBJECTS = {"beard": ("beard", 0.5383, 0.5359), "pair4-c": ("pair4", 0.4907, 0.5958)}
NEG = ("bad quality, worst quality, text, signature, watermark, extra limbs, "
       "asymmetric face, distorted features")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    images = build_images(extract_man_truths(detector))
    report: dict[str, dict] = {}

    for key, (subject, hy_base, qwen_base) in SUBJECTS.items():
        ref = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        truth = Image.open(images[key]["truth"]).convert("RGB")
        prompt = v8_prompt(v8_report(subject))
        w, h = fal_provider.fit_size(ref.width, ref.height)
        url = fal_provider.to_data_url(ref)
        row: dict = {"baseline_qwen_direct_best": qwen_base}
        strip = [("PROFILE (only image sent)", ref)]
        for tag, start_step in (("pulid_start4", 4), ("pulid_start0", 0)):
            path = OUT / f"{key}__{tag}.png"
            try:
                if path.exists():
                    img = Image.open(path).convert("RGB")
                else:
                    img = call_edit(PULID, {
                        "prompt": prompt, "reference_image_url": url, "seed": 101,
                        "id_weight": 1.0, "true_cfg": 1, "start_step": start_step,
                        "guidance_scale": 4, "num_inference_steps": 20,
                        "negative_prompt": NEG,
                        "image_size": {"width": w, "height": h}, "sync_mode": True,
                        "enable_safety_checker": False}, fal_provider.timeout())[0]
                    img.save(path)
                score = vs_image(img, truth, embed)
                row[tag] = {"vs_truth": score, "vs_input": vs_image(img, ref, embed)}
                strip.append((f"{tag}  vs_truth={score}", img))
                print(f"[{key}] {tag}: vs_truth={score} (qwen direct best {qwen_base})", flush=True)
            except Exception as exc:
                print(f"[{key}] {tag} FAILED: {exc}", flush=True)
                row[tag] = {"error": str(exc)[:300]}
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit("ABORT — fal balance exhausted")
        strip.append(("REAL TRUTH (never sent)", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__PULID_BOARD.png")
        report[key] = row

    (OUT / "pulid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
