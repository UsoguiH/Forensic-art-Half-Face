"""Score half-face completion against photos we already hold in full.

Every frontal portrait in ``half photo/`` is cut down the middle, one half is
thrown away, and the pipeline is asked to rebuild it. Because the discarded half
still exists, each run is checkable two ways:

  * **pixel-exactness** — the supplied half must come back byte-for-byte;
    anything else is a hard failure, not a low score.
  * **fidelity** — PSNR/MAE and ArcFace cosine of the *generated* half against
    the real one it never saw.

Run from the repo root::

    python tools/test_half_face.py                     # both halves, every frontal photo
    python tools/test_half_face.py --sides left        # cheaper: one half each
    python tools/test_half_face.py --model klein       # FLUX.2 [klein] 9B instead of [dev]
    python tools/test_half_face.py --no-arcface        # skip the identity scores

Outputs land in ``data/half_face_tests/`` as PNGs plus ``report.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import fal_provider  # noqa: E402
import half_face as hf  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #

def load_detector():
    """InsightFace, if it is installed and its weights are cached."""
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        print(f"[warn] insightface unavailable ({exc}); running without face metrics")
        return None
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as exc:
            print(f"[warn] could not prepare buffalo_l ({exc}); running without face metrics")
            return None
    return app


def detections(app, image: Image.Image) -> list:
    if app is None:
        return []
    try:
        return list(app.get(np.asarray(image.convert("RGB"))[:, :, ::-1]))
    except Exception:
        return []


def biggest(faces: list):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def frontality(face) -> float | None:
    """0 = full profile, 1 = dead-on frontal.

    From the 5-point kps: on a frontal face the nose sits midway between the
    eyes and the eyes are far apart relative to the head box. Turning the head
    breaks both at once, so the smaller of the two signals is the safe one.
    """
    try:
        kps = np.asarray(face.kps, dtype=np.float32)
        x0, _, x1, _ = (float(v) for v in face.bbox[:4])
    except Exception:
        return None
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_gap = float(abs(right_eye[0] - left_eye[0]))
    box = max(1.0, x1 - x0)
    if eye_gap < 1e-3:
        return 0.0
    centred = 1.0 - min(1.0, abs(nose[0] - (left_eye[0] + right_eye[0]) / 2.0) / (eye_gap / 2.0))
    spread = min(1.0, (eye_gap / box) / 0.34)
    return round(min(centred, spread), 3)


def cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if av.shape != bv.shape:
        return None
    an, bn = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if an == 0 or bn == 0:
        return None
    return round(float(np.dot(av / an, bv / bn)), 4)


def embedding(app, image: Image.Image):
    face = biggest(detections(app, image))
    return None if face is None else np.asarray(face.normed_embedding, dtype=np.float32)


# --------------------------------------------------------------------------- #

def make_renderer(model: str | None, seed: int, steps: int | None, guidance: float | None, log: list):
    guidance = hf.DEFAULT_GUIDANCE if guidance is None else guidance

    def render(images, prompt, width, height):
        w, h = fal_provider.fit_size(width, height)
        started = time.time()
        out, slug = fal_provider.edit_images(
            prompt, images, width=w, height=h, seed=seed,
            steps=steps, guidance=guidance, model=model,
        )
        meta = {"provider": "fal", "model": slug, "render_size": [w, h],
                "seconds": round(time.time() - started, 2)}
        log.append(meta)
        return out[0], meta
    return render


def contact_sheet(truth: Image.Image, half: Image.Image, canvas: Image.Image, out: Image.Image) -> Image.Image:
    """truth | supplied half | mirrored start | rebuilt — all at one height."""
    panels = [truth, half, canvas, out]
    height = max(p.height for p in panels)
    scaled = [p.resize((max(1, round(p.width * height / p.height)), height), Image.Resampling.LANCZOS)
              for p in panels]
    gap = 12
    sheet = Image.new("RGB", (sum(p.width for p in scaled) + gap * (len(scaled) - 1), height), (16, 18, 24))
    x = 0
    for panel in scaled:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    return sheet


def run_one(path: Path, side: str, app, renderer, out_dir: Path, args) -> dict:
    truth = Image.open(path).convert("RGB")
    supplied = hf.take_half(truth, side)

    started = time.time()
    result = hf.complete(
        supplied,
        renderer=renderer,
        side=side,
        seed=args.seed,
        tone_match=not args.no_tone_match,
        seam_band=args.seam_band,
        align_radius=args.align_radius,
        align_scale=args.align_scale,
        detector=(lambda img: detections(app, img)) if app is not None else None,
    )
    elapsed = round(time.time() - started, 2)

    rebuilt = result["image"]
    # The canvas is 2x the half, which can differ from the original by a pixel
    # when the source width is odd; compare like with like.
    reference = truth.resize(rebuilt.size, Image.Resampling.LANCZOS)
    accuracy = hf.verify(result, reference)

    # Independent re-check of the guarantee, from the file that will be shipped.
    geo: hf.Geometry = result["geometry"]
    kx0, ky0, kx1, ky1 = geo.keep_box
    exact = np.array_equal(
        np.asarray(rebuilt)[ky0:ky1, kx0:kx1], np.asarray(supplied.convert("RGB"))
    )

    identity = {}
    if app is not None:
        truth_vec = embedding(app, truth)
        identity = {
            "rebuilt_vs_truth": cosine(embedding(app, rebuilt), truth_vec),
            "mirror_vs_truth": cosine(embedding(app, result["canvas"]), truth_vec),
            "half_vs_truth": cosine(embedding(app, supplied), truth_vec),
        }

    stem = f"{path.stem.replace(' ', '_')}__keep-{side}"
    if not args.no_images:
        out_dir.mkdir(parents=True, exist_ok=True)
        supplied.save(out_dir / f"{stem}__1_supplied_half.png")
        result["canvas"].save(out_dir / f"{stem}__2_mirror_start.png")
        rebuilt.save(out_dir / f"{stem}__3_rebuilt.png")
        reference.save(out_dir / f"{stem}__4_truth.png")
        contact_sheet(reference, supplied, result["canvas"], rebuilt).save(
            out_dir / f"{stem}__sheet.png"
        )

    return {
        "file": path.name,
        "kept_side": side,
        "pixel_exact": bool(result["pixel_exact"] and exact),
        "geometry": geo.as_dict(),
        "side_detected_by": result["side_detected_by"],
        "align": result["align"],
        "align_residual": result["align_residual"],
        "seam_step": result["seam_step"],
        "accuracy": accuracy,
        "identity": identity,
        "render": result["render"],
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(ROOT / "half photo"), help="folder of test photos")
    parser.add_argument("--out", default=str(ROOT / "data" / "half_face_tests"), help="where to write results")
    parser.add_argument("--files", nargs="*", default=None, help="specific file names to test")
    parser.add_argument("--sides", default="left,right", help="which half to keep (left,right,top,bottom)")
    parser.add_argument("--model", default=None, help="dev | klein | a full fal slug")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seam-band", type=int, default=None)
    parser.add_argument("--align-radius", type=int, default=6)
    parser.add_argument("--align-scale", type=float, default=0.08,
                        help="how much re-framing the aligner may undo (0 = translation only)")
    parser.add_argument("--frontal-threshold", type=float, default=0.55,
                        help="skip photos below this frontality score (0 = keep all)")
    parser.add_argument("--no-arcface", action="store_true", help="skip face detection and identity scores")
    parser.add_argument("--no-tone-match", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="metrics only, write no PNGs")
    args = parser.parse_args()

    if not fal_provider.available():
        print("FAL_KEY is not set and no key file was found — nothing to render with.")
        return 2

    source = Path(args.dir)
    if args.files:
        photos = [source / name for name in args.files]
    else:
        photos = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not photos:
        print(f"No images found in {source}")
        return 2

    app = None if args.no_arcface else load_detector()

    selected: list[Path] = []
    for path in photos:
        if args.frontal_threshold <= 0 or app is None:
            selected.append(path)
            continue
        face = biggest(detections(app, Image.open(path).convert("RGB")))
        score = frontality(face) if face is not None else None
        if score is None or score < args.frontal_threshold:
            print(f"  skip {path.name}  (frontality {score}) — half-face completion needs a frontal photo")
            continue
        print(f"  use  {path.name}  (frontality {score})")
        selected.append(path)
    if not selected:
        print("No frontal photos matched. Re-run with --frontal-threshold 0 to force.")
        return 2

    sides = [s.strip().lower() for s in args.sides.split(",") if s.strip()]
    for side in sides:
        if side not in hf.SIDES:
            print(f"Unknown side '{side}'; expected one of {hf.SIDES}")
            return 2

    out_dir = Path(args.out)
    log: list = []
    renderer = make_renderer(args.model, args.seed, args.steps, args.guidance, log)

    rows: list[dict] = []
    for path in selected:
        for side in sides:
            label = f"{path.name} [keep {side}]"
            print(f"\n>>> {label}")
            try:
                row = run_one(path, side, app, renderer, out_dir, args)
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                rows.append({"file": path.name, "kept_side": side, "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append(row)
            acc, ident = row["accuracy"], row["identity"]
            print(f"    pixel-exact : {row['pixel_exact']}")
            print(f"    model       : {row['render'].get('model')}  ({row['seconds']}s)")
            print(f"    align       : scale={row['align']['scale']} dy={row['align']['dy']} "
                  f"dx={row['align']['dx']} residual={row['align_residual']}  "
                  f"seam step={row['seam_step']}")
            print(f"    generated   : PSNR {acc['generated_half_psnr']} dB   MAE {acc['generated_half_mae']}")
            if ident:
                print(f"    identity    : rebuilt {ident['rebuilt_vs_truth']} | "
                      f"mirror {ident['mirror_vs_truth']} | half {ident['half_vs_truth']}  (vs truth)")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(source),
        "model_requested": args.model or "dev (fal-ai/flux-2/edit)",
        "seed": args.seed,
        "results": rows,
        "fal_spend": fal_provider.spend(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ok = [r for r in rows if not r.get("error")]
    exact = [r for r in ok if r["pixel_exact"]]
    print("\n" + "=" * 78)
    print(f"{len(ok)}/{len(rows)} completed, {len(exact)}/{len(ok)} pixel-exact on the supplied half")
    if ok:
        psnr = [r["accuracy"]["generated_half_psnr"] for r in ok]
        print(f"generated half PSNR vs the real one: mean {sum(psnr) / len(psnr):.2f} dB "
              f"(min {min(psnr):.2f}, max {max(psnr):.2f})")
        sims = [r["identity"].get("rebuilt_vs_truth") for r in ok if r.get("identity")]
        sims = [s for s in sims if s is not None]
        if sims:
            print(f"ArcFace rebuilt-vs-original: mean {sum(sims) / len(sims):.4f} "
                  f"(min {min(sims):.4f}, max {max(sims):.4f})")
    print(f"images + report.json -> {out_dir}")
    return 0 if ok and len(exact) == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
