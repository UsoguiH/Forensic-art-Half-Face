"""generate_testfile — pure-profile shoot-out on the user's "Test File" folder.

Same provable protocol as generate_pure_profile (ONE profile crop per request,
payload sha256 == the saved __0_REF-SENT.png, truths only ever opened locally),
pointed at the 8 hand-picked profile photos in half photo/Test File.

Man-cluster truths are the frontal panels cut out of the DST collages
(man 1.png / man 2.png) — extracted here, saved beside the results, never sent.

Usage (from Forensic_Art_v8):  python tools/generate_testfile.py [--mock] [--only KEY]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import generate_pure_profile as gpp  # noqa: E402
from iterate_v7 import load_engine  # noqa: E402

HALF = Path(r"D:\Claude\Forensic_ART_Original\Forensic_Art\half photo")
TESTFILE = HALF / "Test File"
DST = HALF / "---DateSet Test---"
OUT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "Model_Compare_TestFile"


def extract_man_truths(detector) -> dict[str, Path]:
    """Cut the frontal right panel out of each man collage → local truth files."""
    OUT.mkdir(parents=True, exist_ok=True)
    truths = {}
    # 151836 is the elevator corridor = man 1.png's left panel; 111011 faces the
    # wall close-up = man 2.png's left panel (matched visually).
    for name, collage in (("man1", DST / "man 1.png"), ("man2", DST / "man 2.png")):
        path = OUT / f"_truth_{name}_frontal_panel.png"
        if not path.exists():
            img = Image.open(collage).convert("RGB")
            crop = gpp.face_crop_by(img, detector, lambda fs: min(fs, key=gpp.profile_ratio))
            crop.save(path)
        truths[name] = path
    return truths


def build_images(truths: dict[str, Path]) -> dict[str, dict]:
    return {
        "beard":     {"file": TESTFILE / "WhatsApp Image 2026-08-02 at 9.44.08 AM.jpeg",
                      "subject": "beard", "truth": HALF / "WhatsApp Image 2026-08-02 at 9.44.07 AM.jpeg"},
        "man-cctv1": {"file": TESTFILE / "Screenshot 2026-08-03 151836.png",
                      "subject": "man", "truth": truths["man1"]},
        "man-cctv2": {"file": TESTFILE / "Screenshot 2026-08-03 111011.png",
                      "subject": "man", "truth": truths["man2"]},
        "pair4-a":   {"file": TESTFILE / "WhatsApp Image 2026-08-02 at 6.38.18 AM (1).jpeg",
                      "subject": "pair4", "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg"},
        "pair4-b":   {"file": TESTFILE / "WhatsApp Image 2026-08-02 at 6.38.18 AM (2).jpeg",
                      "subject": "pair4", "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg"},
        "pair4-c":   {"file": TESTFILE / "WhatsApp Image 2026-08-02 at 6.38.19 AM.jpeg",
                      "subject": "pair4", "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg"},
        "young-cam3": {"file": TESTFILE / "Screenshot 2026-08-04 092418.png",
                       "subject": "young", "truth": DST / "original 6.png"},  # cluster uncertain — check leak line
        "young-cctv": {"file": TESTFILE / "Screenshot 2026-08-04 093224.png",
                       "subject": "young", "truth": DST / "original 6.png"},
    }


def main() -> None:
    detector, _embed = load_engine(False)
    truths = extract_man_truths(detector)
    gpp.IMAGES = build_images(truths)
    gpp.OUT = OUT
    gpp.main()


if __name__ == "__main__":
    main()
