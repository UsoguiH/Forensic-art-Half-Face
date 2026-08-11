"""test_hunyuan_native — does Hunyuan-native prompting fix its identity drift?

Hunyuan-Image-3.0-Instruct edit expects an imperative EDIT INSTRUCTION
("change X, keep Y") and has a built-in reasoning/rewrite pass
(enable_prompt_expansion). Our v8 prompt is a full T2I scene description with
expansion disabled — possibly the worst fit. A/B on the two cleanest Test File
subjects, same REF-SENT inputs as the main benchmark:

  a. v8 scene prompt, expansion ON     (isolates the expansion effect)
  b. native instruct prompt, expansion OFF
  c. native instruct prompt, expansion ON

Baselines (v8 prompt, expansion OFF): beard 0.5383, pair4-c 0.4907.
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
from iterate_v8 import DESC_BEARD, DESC_PAIR4  # noqa: E402
from compare_models_2511 import v8_report, v8_prompt, call_edit, HUNYUAN  # noqa: E402
from generate_testfile import build_images, extract_man_truths  # noqa: E402

TF = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half\Model_Compare_TestFile")
OUT = TF / "hunyuan_native"

NATIVE = (
    "Turn this side-profile photo into a front-facing official ID portrait of the SAME man, "
    "facing the camera directly, both eyes open and looking into the lens, neutral expression, "
    "head-and-shoulders framing on a plain light-gray studio background with soft even frontal "
    "lighting. This is an identity-preserving edit: keep his exact face — the same bone "
    "structure, the same nose shape seen in the profile, the same eyes, eyebrows, lips, ears, "
    "skin tone and texture, the same age, the same facial-hair density and the same hairline "
    "exactly as photographed. Do not beautify, slim, rejuvenate or idealize him; keep natural "
    "asymmetries and skin texture. Identity notes from the photo, every one must hold: {desc}. "
    "Change only the head pose, clothing framing and background; the person himself must stay "
    "identical."
)

# "Perfect" HY3 prompt — follows the official HunyuanImage 3.0 handbook order
# (main subject & scene -> quality & style -> composition & perspective ->
# lighting & atmosphere -> technical parameters), opened with the imperative
# identity-preserving edit instruction the Instruct checkpoint is tuned for.
PERFECT = (
    # instruction + preservation contract
    "Rotate the man in the reference photo to face the camera directly and render a "
    "front-facing official identification portrait of the SAME man. Strict identity-"
    "preserving edit, not a re-imagination: rebuild his face only from what the profile "
    "shows and keep every visible feature exactly — bone structure, nose shape and bridge "
    "as seen in the profile, eye shape and eyelids, eyebrows, lips, ears, jawline, "
    "hairline, facial-hair length and density, skin tone, skin texture and apparent age. "
    "Keep natural asymmetries, moles and shadows under the eyes. Do not beautify, slim, "
    "rejuvenate, groom or idealize him in any way. "
    # main subject & scene (per-subject examiner notes)
    "Subject: {desc}. He looks straight into the lens with both eyes open and a neutral, "
    "relaxed expression, shoulders square to the camera, in the same clothing as the "
    "reference photo. "
    # image quality & style
    "Unretouched documentary realism in the style of a government ID photograph: natural "
    "skin with visible pores and fine hairs, no makeup, no retouching, no glamour "
    "lighting, ordinary real person. "
    # composition & perspective
    "Head-and-shoulders passport framing, face centered, camera exactly at eye level, "
    "85mm portrait lens perspective with no distortion, head occupying about two thirds "
    "of the frame height, both ears equally visible, head perfectly level. "
    # lighting & atmosphere
    "Soft, even, shadow-free frontal studio lighting on a plain light-gray seamless "
    "background, accurate white balance. "
    # technical parameters
    "Sharp focus across the whole face, high detail, realistic photograph, single person, "
    "one face only, portrait orientation."
)

SUBJECTS = {"beard": ("beard", DESC_BEARD, 0.5383), "pair4-c": ("pair4", DESC_PAIR4, 0.4907)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    images = build_images(extract_man_truths(detector))
    results: dict[str, dict] = {}

    for key, (subject, desc, baseline) in SUBJECTS.items():
        ref = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        truth = Image.open(images[key]["truth"]).convert("RGB")
        w, h = fal_provider.fit_size(ref.width, ref.height)
        url = fal_provider.to_data_url(ref)
        scene = v8_prompt(v8_report(subject))
        native = NATIVE.format(desc=desc)
        perfect = PERFECT.format(desc=desc)
        variants = [("a_scene_expandON", scene, True),        # DONE for beard: 0.6077
                    ("b_native_expandOFF", native, False),
                    ("c_native_expandON", native, True),
                    ("d_perfect_expandON", perfect, True)]
        strip = [("REF SENT", ref)]
        row: dict[str, float | None] = {"baseline_scene_expandOFF": baseline}
        for tag, prompt, expand in variants:
            done = OUT / f"{key}__{tag}.png"
            if done.exists():  # resume after top-up: score, don't re-pay
                img = Image.open(done).convert("RGB")
                score = vs_image(img, truth, embed)
                row[tag] = score
                strip.append((f"{tag}  vs_truth={score}", img))
                print(f"[{key}] {tag} (already rendered): vs_truth={score}", flush=True)
                continue
            try:
                img = call_edit(HUNYUAN, {
                    "prompt": prompt, "image_urls": [url], "seed": 101,
                    "image_size": {"width": w, "height": h}, "num_images": 1,
                    "output_format": "png", "sync_mode": True,
                    "enable_prompt_expansion": expand, "enable_safety_checker": False,
                }, fal_provider.timeout())[0]
            except Exception as exc:
                print(f"[{key}] {tag} FAILED: {exc}", flush=True)
                if any(s in str(exc).lower() for s in ("balance", "locked", " 402")):
                    raise SystemExit(f"ABORT — balance: {exc}")
                continue
            score = vs_image(img, truth, embed)
            row[tag] = score
            img.save(OUT / f"{key}__{tag}.png")
            strip.append((f"{tag}  vs_truth={score}", img))
            print(f"[{key}] {tag}: vs_truth={score} (baseline {baseline})", flush=True)
        strip.append(("REAL TRUTH", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD.png")
        results[key] = row

    (OUT / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
