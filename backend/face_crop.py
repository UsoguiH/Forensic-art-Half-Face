"""Head-and-shoulders normalization for scene photos.

A surveillance frame or a casual phone photo hands the pipeline a whole scene
in which the face is a sliver — 0.5-15% of the pixels.  Every consumer
downstream wants the opposite: the reference images FLUX edits from should be
*mostly face* (attention goes where the pixels are), ArcFace wants the face as
big as it can get it, and the output resolution should be spent on the person,
not the corridor.  So before the profile route does anything else, the scene is
reduced to the one crop everything else agrees on: a portrait-aspect
head-and-shoulders window around the detected face.

Margins are anchored on the face box: generous above (ghutra/shemagh and hair
live there), wider below (shoulders and collar carry clothing identity the
target portrait must keep), symmetric sideways.  The window is then grown to
the requested aspect and clamped to the frame — clamping wins over aspect,
because inventing pixels is the renderer's job, not the cropper's.

Small crops are Lanczos-upscaled — plain resampling only.  Anything cleverer
(face-restoration models, generative SR) measurably hallucinates identity and
is Phase B's problem to gate; a soft-but-honest reference beats a sharp lie.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

# Margins as fractions of the detected face box.
_SIDE = 0.85          # each side, of face width
_ABOVE = 0.90         # of face height — headwear and hair
_BELOW = 1.70         # of face height — neck, collar, shoulders

# The reference handed to the renderer should not be a thumbnail: upscale any
# crop narrower than this, but never by more than _MAX_UPSCALE (beyond ~4x,
# resampling adds nothing and the pixels are honest mush).
_MIN_WIDTH = 768
_MAX_UPSCALE = 4.0


def frontality(face: Any) -> float:
    """0 = full profile, 1 = dead-on frontal, from the detector's 5-point kps.

    Used as a hard gate on generated candidates: a render that kept the head
    turned can score deceptively high similarity to the profile reference —
    it is the one failure mode identity ranking cannot see, because it looks
    exactly like fidelity.
    """
    if face is None:
        return 0.0
    try:
        import numpy as np

        kps = np.asarray(face.kps, dtype=np.float32)
        x0, _, x1, _ = (float(v) for v in face.bbox[:4])
        le, re, nose = kps[0], kps[1], kps[2]
    except Exception:
        return 0.0
    gap = float(abs(float(re[0]) - float(le[0])))
    box = max(1.0, x1 - x0)
    if gap < 1e-3:
        return 0.0
    centred = 1.0 - min(1.0, abs(float(nose[0]) - (float(le[0]) + float(re[0])) / 2.0) / (gap / 2.0))
    spread = min(1.0, (gap / box) / 0.34)
    return round(float(min(centred, spread)), 3)


def biggest_face(faces: list[Any]) -> Any | None:
    """The detection with the largest box — the subject, in every test frame."""
    best, area = None, 0.0
    for face in faces or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
        except Exception:
            continue
        a = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if a > area:
            best, area = face, a
    return best


def tight_face_crop(
    image: Image.Image,
    bbox: "tuple[float, float, float, float]",
    *,
    min_width: int = 512,
) -> Image.Image:
    """Close crop of just the head — hair to chin, a whisker of margin.

    Companion reference to ``head_shoulders_crop``: in a far shot even the
    head-and-shoulders window is mostly wall, so the model's attention budget
    still leaks away from the face.  Handing the face by itself as a second
    reference makes the identity pixels impossible to overlook.
    """
    image = image.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = (float(v) for v in bbox)
    fw, fh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    box = (
        int(round(max(0.0, x0 - fw * 0.35))),
        int(round(max(0.0, y0 - fh * 0.55))),      # hairline lives above the box
        int(round(min(float(width), x1 + fw * 0.35))),
        int(round(min(float(height), y1 + fh * 0.30))),
    )
    crop = image.crop(box)
    if crop.width < min_width:
        scale = min(_MAX_UPSCALE, min_width / max(1, crop.width))
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return crop


def head_shoulders_crop(
    image: Image.Image,
    bbox: "tuple[float, float, float, float]",
    *,
    aspect: "tuple[int, int]" = (3, 4),
) -> "tuple[Image.Image, dict[str, Any]]":
    """Portrait head-and-shoulders crop around ``bbox``.

    Returns (crop, meta); meta records where the crop sits in the source frame
    and how much it was upscaled, so results stay traceable to the evidence.
    """
    image = image.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = (float(v) for v in bbox)
    fw, fh = max(1.0, x1 - x0), max(1.0, y1 - y0)

    cx0 = x0 - fw * _SIDE
    cx1 = x1 + fw * _SIDE
    cy0 = y0 - fh * _ABOVE
    cy1 = y1 + fh * _BELOW

    # Grow the short dimension out to the requested aspect, centred on what is
    # already there; never shrink below the margin-derived window.
    want = aspect[0] / aspect[1]                       # width / height
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw / ch < want:                                 # too narrow -> widen
        pad = (ch * want - cw) / 2.0
        cx0, cx1 = cx0 - pad, cx1 + pad
    else:                                              # too squat -> deepen
        pad = (cw / want - ch) / 2.0
        # Faces sit high in a head-and-shoulders frame: grow mostly downward.
        cy0, cy1 = cy0 - pad * 0.35, cy1 + pad * 0.65

    # Clamp by sliding first (preserves size), cropping only at the frame edge.
    cw, ch = cx1 - cx0, cy1 - cy0
    cx0 = min(max(0.0, cx0), max(0.0, width - cw))
    cy0 = min(max(0.0, cy0), max(0.0, height - ch))
    cx1, cy1 = min(float(width), cx0 + cw), min(float(height), cy0 + ch)

    box = (int(round(cx0)), int(round(cy0)), int(round(cx1)), int(round(cy1)))
    crop = image.crop(box)

    scale = 1.0
    if crop.width < _MIN_WIDTH:
        scale = min(_MAX_UPSCALE, _MIN_WIDTH / max(1, crop.width))
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )

    meta = {
        "crop_box": list(box),
        "source_size": [width, height],
        "crop_size": [crop.width, crop.height],
        "upscale": round(scale, 2),
        "face_frac_before": round((fw * fh) / (width * height), 4),
        "face_frac_after": round(
            (fw * fh) / max(1.0, (box[2] - box[0]) * (box[3] - box[1])), 4
        ),
    }
    return crop, meta
