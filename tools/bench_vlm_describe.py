"""bench_vlm_describe — the acceptance gate for VLM auto-descriptions.

Stage 1 (--text-only, ~$0.01): Qwen3-VL writes descriptions for the four known
subjects from the same evidence photos the analyst used; prints them
side-by-side with the hand-written DESC_* for hallucination review.

Stage 2 (default, ~$0.5): strict Test File protocol — same sha-verified single
profile crops, Qwen-Edit-2511 renderer, seeds 101+202 — three prompt arms per
image (hand-written / VLM / no description), scored buffalo_l vs truth.

GATE: VLM mean vs_truth >= hand-written mean - 0.01 and no visible invented
feature on the boards -> flip FACELAB_AUTODESCRIBE default to 1.

Usage (needs FACELAB_VLM_KEY / OPENROUTER_API_KEY / data/openrouter_key.txt):
    python tools/bench_vlm_describe.py --text-only
    python tools/bench_vlm_describe.py
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
import vlm  # noqa: E402
from halfface import build_evidence  # noqa: E402
from halfface import describe as hf_describe  # noqa: E402
from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from iterate_v8 import DESC_BEARD, DESC_PAIR4, DESC_MAN2, DESC_YOUNG_A  # noqa: E402
from compare_models_2511 import v8_report, v8_prompt, call_edit, QWEN  # noqa: E402
from generate_testfile import build_images, extract_man_truths  # noqa: E402

TF = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_TestFile"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "VLM_Describe_Bench"

HAND = {"beard": DESC_BEARD, "pair4-c": DESC_PAIR4, "man-cctv2": DESC_MAN2,
        "young-cctv": DESC_YOUNG_A}
RENDER_KEYS = ("beard", "pair4-c", "man-cctv2")  # stage-2 subset (budget)
SEEDS = (101, 202)


def swap_description(scene_prompt: str, hand: str, new: str) -> str:
    """The v8 report prompt embeds the hand-written description verbatim —
    swap it for the VLM text so every other block stays byte-identical."""
    if hand in scene_prompt:
        return scene_prompt.replace(hand, new)
    raise RuntimeError("hand-written description not found in scene prompt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()

    if not vlm.available():
        raise SystemExit(f"No VLM key yet — {vlm.health()['error']}")
    OUT.mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)
    images = build_images(extract_man_truths(detector))

    # ---- stage 1: descriptions side-by-side --------------------------------
    vlm_desc: dict[str, str] = {}
    for key, hand in HAND.items():
        primary = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        ev = build_evidence(primary, [], detector=detector, embed=embed, who="a man")
        text, meta = hf_describe.from_image(ev)
        vlm_desc[key] = text
        print(f"\n=== {key} (tier={ev.tier}, meta={meta}) ===")
        print(f"--- analyst ({len(hand)} ch):\n{hand}")
        print(f"--- vlm ({len(text)} ch):\n{text or '(EMPTY — fell back)'}")
    (OUT / "descriptions.json").write_text(
        json.dumps({"hand": HAND, "vlm": vlm_desc}, indent=2), encoding="utf-8")
    if args.text_only:
        print(f"\n[bench] text stage done -> {OUT / 'descriptions.json'}")
        return

    # ---- stage 2: render A/B under the strict protocol ---------------------
    subject_of = {"beard": "beard", "pair4-c": "pair4", "man-cctv2": "man"}
    report: dict[str, dict] = {}
    for key in RENDER_KEYS:
        if not vlm_desc.get(key):
            print(f"[{key}] no VLM description — skipping renders", flush=True)
            continue
        ref = Image.open(TF / f"{key}__0_REF-SENT.png").convert("RGB")
        truth = Image.open(images[key]["truth"]).convert("RGB")
        w, h = fal_provider.fit_size(ref.width, ref.height)
        url = fal_provider.to_data_url(ref)
        # always the forensic/std prompt — the picked style can be the
        # idportrait control, which embeds no description to swap
        scene = v8_report(subject_of[key])["prompts"]["forensic/std"]
        hand = HAND[key]
        # nodesc = the same prompt minus ONLY the identity-notes block; the
        # anti-idealization block ("This is an ordinary...") must survive.
        i_id = scene.find("Identity notes")
        i_anti = scene.find("This is an ordinary")
        nodesc = (scene[:i_id] + scene[i_anti:]).strip() if 0 <= i_id < i_anti else scene
        arms = {
            "hand": scene,
            "vlm": swap_description(scene, hand, vlm_desc[key]),
            "nodesc": nodesc,
        }
        row: dict[str, dict] = {}
        strip = [("PROFILE", ref)]
        for tag, prompt in arms.items():
            scores = []
            best = None
            for seed in SEEDS:
                path = OUT / f"{key}__{tag}__seed{seed}.png"
                try:
                    img = (Image.open(path).convert("RGB") if path.exists()
                           else call_edit(QWEN, {
                               "prompt": prompt, "image_urls": [url], "seed": seed,
                               "image_size": {"width": w, "height": h},
                               "num_images": 1, "output_format": "png",
                               "sync_mode": True, "enable_safety_checker": False,
                               "acceleration": "regular"}, fal_provider.timeout())[0])
                    img.save(path)
                    s = vs_image(img, truth, embed)
                    scores.append(s)
                    if best is None or (s or -1) > (best[0] or -1):
                        best = (s, img)
                except Exception as exc:
                    print(f"[{key}] {tag} seed{seed} FAILED: {exc}", flush=True)
                    if any(t in str(exc).lower() for t in ("balance", "locked", " 402")):
                        raise SystemExit("ABORT — fal balance exhausted")
            if best:
                row[tag] = {"scores": scores, "best": best[0]}
                strip.append((f"{tag}  best={best[0]}", best[1]))
                print(f"[{key}] {tag}: seeds={scores} best={best[0]}", flush=True)
        strip.append(("REAL TRUTH", truth))
        label_strip(strip, height=560).save(OUT / f"{key}__BOARD.png")
        report[key] = row

    hands = [r["hand"]["best"] for r in report.values() if "hand" in r]
    vlms = [r["vlm"]["best"] for r in report.values() if "vlm" in r]
    gate = {
        "hand_mean": round(sum(hands) / len(hands), 4) if hands else None,
        "vlm_mean": round(sum(vlms) / len(vlms), 4) if vlms else None,
    }
    gate["pass"] = (gate["hand_mean"] is not None and gate["vlm_mean"] is not None
                    and gate["vlm_mean"] >= gate["hand_mean"] - 0.01)
    report["_gate"] = gate
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
