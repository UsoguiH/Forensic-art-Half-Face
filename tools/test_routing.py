"""Offline routing battery: does classify() send every upload down the right path?

No renders, no network — detector + classifier only. Three suites:

  * New DataSet CCTV panels (paired sheets are split; left panel only) —
    every one is a far/profile subject and must route "profile", never the
    mirror pipeline (the pre-fix behaviour sent 7/16 to "half").
  * Old close-up profiles (WhatsApp) — must stay "profile"; their frontal
    counterparts must stay "full".
  * Derived halves — take_half() of the old frontals must still route "half"
    (the pixel-exact pipeline's contract is untouched).

Run from the repo root:  python tools/test_routing.py
Exit code 0 = every expectation held.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import half_face as hf  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_detector():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return lambda img: list(app.get(np.asarray(img.convert("RGB"))[:, :, ::-1]))


def left_panel(image: Image.Image) -> Image.Image:
    """Paired review sheets are input|target; keep only the input panel."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    cols = np.abs(np.diff(arr, axis=1)).mean(axis=(0, 2))
    w = len(cols)
    lo, hi = int(w * 0.30), int(w * 0.70)
    k = lo + int(np.argmax(cols[lo:hi]))
    if cols[k] > 6.0 * np.median(cols):
        return image.crop((0, 0, k, image.height))
    return image


def main() -> int:
    detector = load_detector()
    failures: list[str] = []
    checked = 0

    def expect(label: str, image: Image.Image, want: set[str]):
        nonlocal checked
        checked += 1
        kind, why = hf.classify(image, detector)
        ok = kind in want
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {label:<55} -> {kind:<8} ({why.get('reason')})")
        if not ok:
            failures.append(f"{label}: got {kind}, wanted {sorted(want)}")

    print("== New DataSet (CCTV): every subject must route 'profile'")
    for path in sorted((ROOT / "half photo" / "New DataSet").iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        expect(path.name, left_panel(Image.open(path).convert("RGB")), {"profile"})

    print("== Old close-ups: profiles stay 'profile', frontals stay 'full'")
    old = ROOT / "half photo"
    for name in (
        "WhatsApp Image 2026-08-02 at 6.38.18 AM.jpeg",
        "WhatsApp Image 2026-08-02 at 6.38.18 AM (1).jpeg",
        "WhatsApp Image 2026-08-02 at 6.38.18 AM (2).jpeg",
    ):
        expect(name, Image.open(old / name).convert("RGB"), {"profile"})
    frontals = [
        "WhatsApp Image 2026-08-02 at 6.38.17 AM.jpeg",
        "WhatsApp Image 2026-08-02 at 6.38.17 AM (1).jpeg",
        "WhatsApp Image 2026-08-02 at 6.38.17 AM (2).jpeg",
        # (3) is a dead-frontal CCTV frame, verified by eye — "full" is correct.
        "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg",
    ]
    for name in frontals:
        expect(name, Image.open(old / name).convert("RGB"), {"full"})

    print("== Derived halves: the pixel-exact contract must be untouched")
    for name in frontals:
        image = Image.open(old / name).convert("RGB")
        for side in ("left", "right"):
            expect(f"{name} [take {side}]", hf.take_half(image, side), {"half"})

    print()
    if failures:
        print(f"{len(failures)}/{checked} FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all {checked} routing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
