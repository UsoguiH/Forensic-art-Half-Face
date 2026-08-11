"""classify_new_dataset — sort "New DataSet" files into PAIRED / UNPAIRED (2026-08-10).

PAIRED  = one file holding two panels: side profile + true frontal of the same
          person. Split at the seam between the two face boxes; profile panel
          becomes <key>__input.png, frontal panel <key>__target.png.
UNPAIRED = profile only, no reference — copied to unpaired/ as inference targets.

Every decision is printed (face count, profile_ratios, seam, panel identity
cosine) and each paired split gets a board image (original | input | target)
so the split can be verified by eye, per the goal spec.

Usage (from Forensic_Art_v8):  python tools/classify_new_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

from halfface.rank import vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine  # noqa: E402
from generate_pure_profile import profile_ratio  # noqa: E402

DATASET = Path(r"D:\Claude\Forensic_ART_Original\Forensic_Art\half photo\New DataSet")
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "NewDataSet_Split"

# a second face only counts as a panel if it is at least this fraction of the
# biggest face's height. Kept LOW on purpose: several collages pair a large
# frontal with a FAR-AWAY profile shot (face 83px vs 555px, ratio 0.15) —
# a 0.45 gate silently reclassified 7 real pairs as unpaired. Pose + horizontal
# separation do the real filtering; boards catch any background-face slip-ups.
MIN_REL_FACE = 0.12


def significant_faces(faces) -> list:
    if not faces:
        return []
    big = max((f.bbox[3] - f.bbox[1]) for f in faces)
    return [f for f in faces if (f.bbox[3] - f.bbox[1]) >= MIN_REL_FACE * big
            and f.det_score >= 0.5]


def classify(img: Image.Image, faces) -> dict:
    """Decide PAIRED vs UNPAIRED and, for paired, where the seam is."""
    sig = significant_faces(faces)
    info = {"n_faces_all": len(faces or []), "n_faces_sig": len(sig),
            "ratios": [round(profile_ratio(f), 2) for f in sig]}
    if len(sig) < 2:
        info["kind"] = "unpaired"
        return info

    # two biggest significant faces, ordered left→right
    a, b = sorted(sorted(sig, key=lambda f: -(f.bbox[3] - f.bbox[1]))[:2],
                  key=lambda f: f.bbox[0])
    gap_l, gap_r = float(a.bbox[2]), float(b.bbox[0])
    info["h_separated"] = gap_r >= gap_l - 0.05 * img.width
    if not info["h_separated"]:
        info["kind"] = "unpaired"
        info["note"] = "two faces but horizontally overlapping — not a two-panel collage"
        return info

    ra, rb = profile_ratio(a), profile_ratio(b)
    info["left_ratio"], info["right_ratio"] = round(ra, 2), round(rb, 2)
    if max(ra, rb) < 0.55:
        info["kind"] = "unpaired"
        info["note"] = "two frontal-ish faces — no profile panel, not our pair pattern"
        return info

    info["kind"] = "paired"
    info["seam_x"] = int((gap_l + gap_r) / 2)
    info["profile_side"] = "left" if ra > rb else "right"
    return info


def main() -> None:
    (OUT / "paired").mkdir(parents=True, exist_ok=True)
    (OUT / "unpaired").mkdir(parents=True, exist_ok=True)
    (OUT / "boards").mkdir(parents=True, exist_ok=True)
    detector, embed = load_engine(False)

    report: dict[str, dict] = {}
    files = sorted(p for p in DATASET.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    for path in files:
        key = path.stem.replace(" ", "_")
        img = Image.open(path).convert("RGB")
        info = classify(img, detector(img))
        info["file"] = path.name
        info["size"] = list(img.size)

        if info["kind"] == "paired":
            seam = info["seam_x"]
            left, right = img.crop((0, 0, seam, img.height)), img.crop((seam, 0, img.width, img.height))
            inp, tgt = (left, right) if info["profile_side"] == "left" else (right, left)
            inp.save(OUT / "paired" / f"{key}__input.png")
            tgt.save(OUT / "paired" / f"{key}__target.png")
            info["panel_identity_cos"] = vs_image(inp, tgt, embed)
            # sanity: the target panel must contain a MORE frontal face than the input
            tf, inf_ = detector(tgt), detector(inp)
            info["target_ratio_check"] = round(min((profile_ratio(f) for f in tf), default=9.9), 2)
            info["input_ratio_check"] = round(max((profile_ratio(f) for f in inf_), default=-1.0), 2)
            label_strip([("ORIGINAL FILE", img), ("INPUT (profile)", inp),
                         ("TARGET (true frontal)", tgt)], height=520
                        ).save(OUT / "boards" / f"{key}__split.png")
        else:
            img.save(OUT / "unpaired" / f"{key}.png")

        report[key] = info
        print(f"[{key}] {info['kind'].upper():8s} faces={info['n_faces_sig']}/{info['n_faces_all']} "
              f"ratios={info['ratios']} "
              + (f"seam={info['seam_x']} profile={info['profile_side']} "
                 f"panel_cos={info['panel_identity_cos']} "
                 f"tgt_frontal_ratio={info['target_ratio_check']} inp_profile_ratio={info['input_ratio_check']}"
                 if info["kind"] == "paired" else info.get("note", "")), flush=True)

    n_pair = sum(1 for v in report.values() if v["kind"] == "paired")
    print(f"\nTOTAL: {len(report)} files → {n_pair} paired, {len(report) - n_pair} unpaired")
    (OUT / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report → {OUT / 'classification_report.json'}")


if __name__ == "__main__":
    main()
