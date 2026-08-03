"""Pixel-exact half-face completion.

Give it one half of a portrait and it returns the whole face, with a hard
guarantee: **every pixel of the half you supplied is byte-for-byte identical in
the output**.  Only the missing half is synthesised.

Why the guarantee needs its own module.  A diffusion edit always repaints the
whole canvas — even the part you asked it to leave alone — so the "kept" half
comes back subtly resampled, re-lit and re-toned.  For forensic use that is
worse than useless: the one region you actually have evidence for is the region
you can no longer trust.  So the model is used only as a *generator for the
missing side*, and the answer is assembled here:

  1. build a full canvas from the supplied half plus a mirrored pre-fill, so the
     model starts from correct structure, skin tone, hair and background;
  2. render it with FLUX.2 (``renderer``), the bare half riding along as an
     identity anchor;
  3. put the render back in register with the supplied half — scale *and*
     translation, both measured rather than assumed, because a model pushed hard
     enough to genuinely repaint also re-frames, and a drifted seam is a visible
     seam;
  4. tone-match the generated half, then fade its seam-adjacent strip into the
     mirror, which is exactly continuous with the pixels about to be pasted back;
  5. paste the supplied half over the result at its original coordinates.

Everything before step 5 only ever touches pixels on the generated side, so the
guarantee is structural rather than best-effort.  ``complete`` still checks it
and reports ``pixel_exact``; if the check ever failed the result would be a lie,
so it raises instead of returning.

The module is orientation-agnostic: left, right, top or bottom halves all work.
It has no provider knowledge — pass any ``renderer``; ``backend.py`` supplies
one backed by fal.ai FLUX.2 [dev].

None of this applies to a *side-profile* photograph: that is a complete head,
not half of a frontal one, and mirroring it produces a two-faced canvas.
``classify`` tells the two apart so the caller can frontalize profiles instead
of running this pipeline on them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

SIDES = ("left", "right", "top", "bottom")

# Measured on the frontal photos in ``half photo/`` (see tools/test_half_face.py):
# ArcFace similarity of the rebuilt face to the ground truth averaged 0.59 at
# guidance 1.0-1.5 against 0.54 at 2.0-2.5, and low guidance also cut how far the
# model re-framed the canvas, which is what puts the seam at risk. 1.5 rather
# than 1.0 because 1.0 occasionally bolted and re-cropped by 4%.
DEFAULT_GUIDANCE = 1.5

# A column/row is "blank" when it carries essentially no detail — that is what a
# masked-out half looks like, whatever colour it was filled with.
_BLANK_STD = 2.0
_BLANK_MIN_FRACTION = 0.12

Renderer = Callable[[list[Image.Image], str, int, int], "tuple[Image.Image, dict[str, Any]]"]


@dataclass(frozen=True)
class Geometry:
    """Where the supplied half sits inside the completed canvas."""

    side: str                       # which half was SUPPLIED
    axis: str                       # "x" for left/right, "y" for top/bottom
    keep_first: bool                # supplied half occupies the low indices
    size: tuple[int, int]           # full canvas (width, height)
    keep_box: tuple[int, int, int, int]
    gen_box: tuple[int, int, int, int]

    @property
    def seam(self) -> int:
        """Index along ``axis`` where supplied and generated content meet."""
        i = 0 if self.axis == "x" else 1
        return self.keep_box[i + 2] if self.keep_first else self.keep_box[i]

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "axis": self.axis,
            "seam": self.seam,
            "width": self.size[0],
            "height": self.size[1],
            "keep_box": list(self.keep_box),
            "gen_box": list(self.gen_box),
        }


# --------------------------------------------------------------------------- #
# oriented views: all seam math is written once, for "kept on the left"
# --------------------------------------------------------------------------- #

def _view(arr: np.ndarray, geo: Geometry) -> np.ndarray:
    """A *view* of ``arr`` with the supplied half in the leading columns.

    Transposes and reverse-slices are numpy views, so writes through the view
    land in the original array — which is exactly what the seam repair needs.
    """
    v = arr if geo.axis == "x" else arr.transpose(1, 0, 2)
    return v if geo.keep_first else v[:, ::-1]


def _mirror(arr: np.ndarray, axis: str) -> np.ndarray:
    return arr[:, ::-1] if axis == "x" else arr[::-1, :]


def _filler_for(keep: np.ndarray, geo: Geometry) -> np.ndarray:
    """The supplied half mirrored, sized to fit the generated box exactly.

    Cropping a half gives two halves of equal size, but a *masked* canvas of odd
    width leaves the generated side one pixel wider, and there is no mirror
    source for that pixel. The mirror is therefore anchored at the seam and the
    far edge replicated — a background column, rather than the shape mismatch
    that used to make tone-matching and the seam feather silently no-op.
    """
    filler = _mirror(keep, geo.axis)
    gx0, gy0, gx1, gy1 = geo.gen_box
    want = (gx1 - gx0) if geo.axis == "x" else (gy1 - gy0)
    lane = 1 if geo.axis == "x" else 0
    have = filler.shape[lane]
    if have == want:
        return filler
    if have > want:
        cut = slice(0, want) if geo.keep_first else slice(have - want, have)
        return filler[:, cut] if lane == 1 else filler[cut]
    before, after = (0, want - have) if geo.keep_first else (want - have, 0)
    pad = ((0, 0), (before, after), (0, 0)) if lane == 1 else ((before, after), (0, 0), (0, 0))
    return np.pad(filler, pad, mode="edge")


# --------------------------------------------------------------------------- #
# input analysis
# --------------------------------------------------------------------------- #

def _blank_run(arr: np.ndarray, axis: str) -> tuple[str, int] | None:
    """Detect a flat, edge-anchored band — a half that was masked out rather
    than cropped away.  Returns (side_supplied, seam) or None."""
    lanes = arr.transpose(1, 0, 2) if axis == "x" else arr        # index 0 walks the axis
    detail = lanes.reshape(lanes.shape[0], -1).std(axis=1)
    flat = detail < _BLANK_STD
    n = len(flat)
    if flat.all():
        # Nothing but flat: a blank image, not a half with a masked side.
        return None
    lead = int(np.argmin(flat))
    tail = n - int(np.argmin(flat[::-1]))
    minimum = max(8, int(n * _BLANK_MIN_FRACTION))

    def solid(lo: int, hi: int) -> bool:
        # Per-lane flatness alone would also accept a smooth background gradient.
        # A masked-out region is one fill colour, so the whole block is flat too.
        return float(lanes[lo:hi].std()) < _BLANK_STD

    if lead >= minimum and lead >= (n - tail) and solid(0, lead):
        # blank at the start -> the supplied half is the high side
        return ("right" if axis == "x" else "bottom"), lead
    if (n - tail) >= minimum and solid(tail, n):
        return ("left" if axis == "x" else "top"), tail
    return None


def _face_score(detections: list[Any], size: tuple[int, int]) -> float:
    """How much a candidate canvas looks like ONE big, centred face."""
    width, height = size
    best = 0.0
    for face in detections or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
            score = float(getattr(face, "det_score", 1.0) or 1.0)
        except Exception:
            continue
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0:
            continue
        centred = 1.0 - min(1.0, abs(((x0 + x1) / 2) / max(1, width) - 0.5) * 2.0)
        best = max(best, score * centred * math.sqrt(area / (width * height)))
    return best


def _edge_energy(arr: np.ndarray, axis: str) -> tuple[float, float]:
    """Detail near each edge along ``axis``. The facial midline (the cut) is busy
    with nose, lip and chin structure; the outer edge is mostly background."""
    lanes = arr.transpose(1, 0, 2) if axis == "x" else arr
    n = lanes.shape[0]
    band = max(2, n // 20)
    gray = lanes.reshape(n, -1).astype(np.float32)
    grad = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return float(grad[:band].mean()), float(grad[-band:].mean())


def analyze(
    image: Image.Image,
    *,
    side: str | None = None,
    axis: str | None = None,
    detector: Callable[[Image.Image], list[Any]] | None = None,
) -> tuple[Geometry, Image.Image, str]:
    """Work out which half was supplied and where the seam goes.

    Returns (geometry, the supplied half as its own image, how it was decided).
    """
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    side = (side or "").strip().lower() or None
    if side in ("auto", ""):
        side = None
    if side and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES} (or auto).")
    axis = (axis or ("x" if side in (None, "left", "right") else "y")).lower()
    if side:
        axis = "x" if side in ("left", "right") else "y"

    # A masked-out half is unambiguous, so look for it first, on both axes.
    for candidate_axis in ((axis,) if side else ("x", "y")):
        found = _blank_run(arr, candidate_axis)
        if not found:
            continue
        blank_side, seam = found
        if side and blank_side != side:
            continue
        geo = _geometry_from_canvas(blank_side, candidate_axis, seam, (width, height))
        return geo, image.crop(geo.keep_box), "masked-canvas"

    # Otherwise the image *is* the half: the cut edge becomes the seam and the
    # canvas doubles along the axis.
    if side is None:
        side, how = _guess_side(image, arr, axis, detector)
    else:
        how = "requested"
    geo = _geometry_from_half((width, height), side)
    return geo, image, how


def _guess_side(
    image: Image.Image,
    arr: np.ndarray,
    axis: str,
    detector: Callable[[Image.Image], list[Any]] | None,
) -> tuple[str, str]:
    """Which edge of a bare half carries the facial midline.

    The reliable test is constructive: mirror the half both ways and ask the
    face detector which canvas is a face.  Joining the *outer* edges produces
    two half-faces looking away from each other, which scores far worse.
    """
    low, high = ("left", "right") if axis == "x" else ("top", "bottom")
    if detector is not None:
        scores: dict[str, float] = {}
        for candidate in (low, high):
            geo = _geometry_from_half(image.size, candidate)
            canvas = build_canvas(image, geo)
            try:
                scores[candidate] = _face_score(detector(canvas), canvas.size)
            except Exception:
                scores[candidate] = 0.0
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best, f"detector({scores[low]:.3f} vs {scores[high]:.3f})"

    lead, tail = _edge_energy(arr, axis)
    # Busier edge == the cut through the face; the half lies on the other side.
    return (low if tail >= lead else high), f"edge-detail({lead:.1f} vs {tail:.1f})"


def classify(
    image: Image.Image,
    detector: Callable[[Image.Image], list[Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """Is this upload really a *half*, or a complete head that must not be mirrored?

    The mirror pipeline assumes the input is one half of a frontal portrait.  Feed
    it a side-profile photo and the mirror step produces a two-faced canvas, which
    the edit model then faithfully keeps.  So before any mirroring happens the
    input is classified from the face detector's five keypoints:

    - ``"half"``     genuine half: a masked-out band, a face clipped at the image
                     border, or nothing detectable — the pixel-exact pipeline runs.
    - ``"profile"``  a complete head seen from the side (tiny interocular span, or
                     the nose far off the eye midpoint) — must be frontalized, not
                     mirrored.
    - ``"full"``     a complete frontal face well inside the frame — there is no
                     missing half to generate.
    """
    arr = np.asarray(image.convert("RGB"))
    for axis in ("x", "y"):
        if _blank_run(arr, axis):
            return "half", {"reason": "masked-canvas"}
    if detector is None:
        return "half", {"reason": "no-detector"}
    try:
        faces = detector(image) or []
    except Exception:
        return "half", {"reason": "detector-failed"}

    width, height = image.size
    best, area = None, 0.0
    for face in faces:
        try:
            x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
        except Exception:
            continue
        a = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if a > area:
            best, area = face, a
    if best is None or area < width * height * 0.02:
        return "half", {"reason": "no-face"}
    kps = getattr(best, "kps", None)
    if kps is None or len(kps) < 3:
        return "half", {"reason": "no-keypoints"}

    x0, y0, x1, y1 = (float(v) for v in best.bbox[:4])
    le, re, nose = (np.asarray(p, dtype=float) for p in kps[:3])
    eye_span = abs(float(re[0] - le[0]))
    eye_frac = eye_span / max(1.0, x1 - x0)
    nose_off = abs(float(nose[0]) - (float(le[0]) + float(re[0])) / 2.0) / max(eye_span, 1e-3)
    stats = {"eye_span_frac": round(eye_frac, 3), "nose_offset": round(nose_off, 2)}

    # A face whose box hugs an image border IS a half: the cut runs through it.
    if x0 <= width * 0.04 or x1 >= width * 0.96 or y0 <= height * 0.04 or y1 >= height * 0.96:
        return "half", {"reason": "face-at-border", **stats}
    # Frontal faces measure eye_frac ≈ 0.38-0.48 and nose_off < 0.25; a head
    # turned far enough that mirroring is nonsense collapses the eye span and
    # throws the nose sideways long before those bounds are approached.
    if eye_frac < 0.30 or nose_off > 0.60:
        return "profile", {"reason": "side-view-pose", **stats}
    return "full", {"reason": "frontal-face-inside-frame", **stats}


def _geometry_from_half(half_size: tuple[int, int], side: str) -> Geometry:
    w, h = half_size
    if side in ("left", "right"):
        size = (w * 2, h)
        keep = (0, 0, w, h) if side == "left" else (w, 0, w * 2, h)
        gen = (w, 0, w * 2, h) if side == "left" else (0, 0, w, h)
        return Geometry(side, "x", side == "left", size, keep, gen)
    size = (w, h * 2)
    keep = (0, 0, w, h) if side == "top" else (0, h, w, h * 2)
    gen = (0, h, w, h * 2) if side == "top" else (0, 0, w, h)
    return Geometry(side, "y", side == "top", size, keep, gen)


def _geometry_from_canvas(side: str, axis: str, seam: int, size: tuple[int, int]) -> Geometry:
    width, height = size
    if axis == "x":
        keep = (0, 0, seam, height) if side == "left" else (seam, 0, width, height)
        gen = (seam, 0, width, height) if side == "left" else (0, 0, seam, height)
        return Geometry(side, "x", side == "left", size, keep, gen)
    keep = (0, 0, width, seam) if side == "top" else (0, seam, width, height)
    gen = (0, seam, width, height) if side == "top" else (0, 0, width, seam)
    return Geometry(side, "y", side == "top", size, keep, gen)


# --------------------------------------------------------------------------- #
# canvas construction
# --------------------------------------------------------------------------- #

def build_canvas(half: Image.Image, geo: Geometry) -> Image.Image:
    """Supplied half plus its mirror image, filling the whole canvas.

    The mirror is only a starting point, but it is a very good one: it hands the
    model the right head size, skin tone, hair mass, collar and background, so
    the edit becomes "make this side natural" instead of "invent a face".
    """
    keep = np.asarray(half.convert("RGB"))
    gx0, gy0, gx1, gy1 = geo.gen_box
    kx0, ky0, kx1, ky1 = geo.keep_box
    canvas = np.zeros((geo.size[1], geo.size[0], 3), dtype=np.uint8)
    canvas[ky0:ky1, kx0:kx1] = keep
    canvas[gy0:gy1, gx0:gx1] = _filler_for(keep, geo)
    return Image.fromarray(canvas)


def prompt_for(geo: Geometry, *, gender: str = "", extra: str = "", anchor: bool = True) -> str:
    """Instruction for the edit model, in the canvas's own orientation.

    Two things had to be said at once, and saying either one alone fails.
    *Repaint*: told only to preserve, FLUX.2 hands the mirrored canvas straight
    back — reversed background text and all.  *Do not re-frame*: told only to
    repaint, it zooms and re-crops, which slides the generated half out of
    register with the half we are going to paste back.

    Note what is absent: no mention of a "seam" or a "line down the middle".
    Naming them made the model draw one.

    ``anchor`` adds the sentence that pairs with sending the bare half as a
    second reference image; that combination measurably held the identity
    closer and halved the re-framing.
    """
    kept = geo.side
    missing = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[kept]
    who = (gender or "").strip()
    who = f"This is a photograph of {who}. " if who else ""
    body = (
        f"{who}The {missing} half of this picture is not real: it was made by mirroring the "
        f"{kept} half, so the face is impossibly symmetric and the background on the {missing} "
        f"side is a reversed copy. Repaint the whole {missing} half so it becomes the genuine "
        f"{missing} side of this same person — their real {missing} eye, eyebrow, eyelid, "
        "nostril, cheek, jawline, ear, hairline and hair, and the real background and clothing "
        "on that side. A real face is never symmetric, so it must differ naturally from the "
        f"{kept} side. "
        f"Keep the {kept} half unchanged. Do not zoom in or out, do not crop and do not "
        "re-frame: the head stays exactly the same size and in exactly the same place, with the "
        "eye line, hairline and shoulders at the same height as they are now. Same person, same "
        "age, same skin tone and skin texture, same moles and marks, same facial hair, same "
        "haircut, same clothing, same camera, same lens, same lighting and exposure. "
        "Photorealistic identification photograph, neutral expression, no beautification, no "
        "makeup, no smoothing."
    )
    if anchor:
        body += (
            " Image 2 is the real, unaltered photograph of this person; keep their identity, "
            "bone structure and features exactly as they are in image 2."
        )
    extra = (extra or "").strip()
    return f"{body} {extra}".strip()


# --------------------------------------------------------------------------- #
# alignment, tone, seam
# --------------------------------------------------------------------------- #

def _gray(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate with edge replication, so no black border is introduced."""
    if dy == 0 and dx == 0:
        return arr
    pad = max(abs(dy), abs(dx))
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = arr.shape[:2]
    y0, x0 = pad - dy, pad - dx
    return padded[y0:y0 + h, x0:x0 + w]


def _scale_about_centre(arr: np.ndarray, scale: float) -> np.ndarray:
    """Zoom around the centre, keeping the canvas size — the inverse of the
    re-framing an edit model applies when it decides to crop in a little."""
    if abs(scale - 1.0) < 1e-4:
        return arr
    height, width = arr.shape[:2]
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    scaled = np.asarray(
        Image.fromarray(arr).resize((sw, sh), Image.Resampling.LANCZOS)
    )
    if scale > 1.0:                       # crop back to size
        x0, y0 = (sw - width) // 2, (sh - height) // 2
        return scaled[y0:y0 + height, x0:x0 + width]
    pad_x, pad_y = (width - sw) // 2, (height - sh) // 2
    out = np.pad(
        scaled,
        ((pad_y, height - sh - pad_y), (pad_x, width - sw - pad_x), (0, 0)),
        mode="edge",
    )
    return out


def align(
    render: np.ndarray,
    keep: np.ndarray,
    geo: Geometry,
    radius: int = 6,
    scale_range: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Put the render back in register with the half we already have.

    A diffusion edit drifts a few pixels and, when pushed hard enough to
    actually repaint, re-frames by a few percent as well.  Either one slides the
    generated half out of alignment with the pasted-back original, which is
    exactly where a seam becomes visible.  The half we hold is a free ground
    truth, so the transform is *measured* rather than assumed: search scale and
    translation for whatever best reproduces it, coarse then fine.

    Returns the corrected render, the transform applied, and the residual.
    """
    kx0, ky0, kx1, ky1 = geo.keep_box
    target = _gray(keep)[::3, ::3]

    def score(candidate: np.ndarray, dy: int, dx: int) -> float:
        moved = _shift(candidate, dy, dx)[ky0:ky1, kx0:kx1]
        return float(np.abs(_gray(moved)[::3, ::3] - target).mean())

    def search(scales, ys, xs, cache) -> tuple[float, int, int, float]:
        best = (1.0, 0, 0, float("inf"))
        for scale in scales:
            key = round(scale, 4)
            if key not in cache:
                cache[key] = _scale_about_centre(render, key)
            candidate = cache[key]
            for dy in ys:
                for dx in xs:
                    error = score(candidate, dy, dx)
                    if error < best[3]:
                        best = (key, dy, dx, error)
        return best

    cache: dict[float, np.ndarray] = {}
    step = max(1, radius // 2)
    coarse_scales = [1.0] if scale_range <= 0 else [
        round(1.0 + scale_range * k / 4.0, 4) for k in range(-4, 5)
    ]
    coarse_shifts = list(range(-radius, radius + 1, step))
    scale, dy, dx, error = search(coarse_scales, coarse_shifts, coarse_shifts, cache)

    fine_scales = sorted({round(scale + d, 4) for d in (-0.01, -0.005, 0.0, 0.005, 0.01)})
    scale, dy, dx, error = search(
        fine_scales,
        range(dy - step, dy + step + 1),
        range(dx - step, dx + step + 1),
        cache,
    )

    corrected = _shift(cache.get(scale, _scale_about_centre(render, scale)), dy, dx)
    return corrected, {"scale": scale, "dy": dy, "dx": dx}, float(error)


def _tone_match(generated: np.ndarray, keep: np.ndarray, geo: Geometry) -> np.ndarray:
    """Pull the generated half onto the supplied half's exposure and colour.

    Compared against the *mirror* of the supplied half, which is the closest
    thing to matching content that exists.  Gains stay tight on purpose — this
    corrects drift, it does not restyle.
    """
    reference = _filler_for(keep, geo).astype(np.float32)
    if reference.shape != generated.shape:
        return generated
    out = generated.astype(np.float32)
    for channel in range(3):
        g, r = out[..., channel], reference[..., channel]
        g_std, r_std = float(g.std()), float(r.std())
        gain = min(1.09, max(0.92, (r_std / g_std) if g_std > 1e-3 else 1.0))
        offset = min(14.0, max(-14.0, float(r.mean()) - gain * float(g.mean())))
        out[..., channel] = g * gain + offset
    return np.clip(out, 0, 255).astype(np.uint8)


def feather_to_mirror(generated: np.ndarray, keep: np.ndarray, geo: Geometry, band: int) -> np.ndarray:
    """Hand the seam-adjacent strip back to the mirror, fading into the render.

    Right at the facial midline a human face really is near-symmetric, and it is
    also where we hold no evidence at all about the other side — so the mirror is
    both the best estimate available and, unlike the render, *exactly* continuous
    with the pixels we are about to paste back.  Fading it out over ``band``
    columns buys a mathematically seamless join for a strip a couple of percent
    of the face wide, and asymmetry still comes from the model everywhere else.

    An earlier version instead measured the step across the seam and corrected
    the render by it.  That fails badly when the render puts an artefact right at
    the join: the probe reads the artefact, and the "correction" smears it into a
    bright band.  Anchoring on the mirror has no such failure mode.

    Only generated pixels are touched, so pixel-exactness is unaffected.
    """
    filler = _filler_for(keep, geo).astype(np.float32)
    if filler.shape != generated.shape:
        return generated
    extent = generated.shape[1] if geo.axis == "x" else generated.shape[0]
    band = int(max(0, min(band, extent)))
    if band == 0:
        return generated

    t = np.arange(band, dtype=np.float32) / band
    alpha = np.ones(extent, dtype=np.float32)
    alpha[:band] = t * t * (3.0 - 2.0 * t)      # smoothstep: flat at both ends
    if not geo.keep_first:
        # The seam is at the far end when the supplied half has the high indices.
        alpha = alpha[::-1]
    weights = alpha.reshape((1, extent, 1) if geo.axis == "x" else (extent, 1, 1))

    blended = generated.astype(np.float32) * weights + filler * (1.0 - weights)
    return np.clip(blended, 0, 255).astype(np.uint8)


def seam_step(canvas: np.ndarray, geo: Geometry) -> float:
    """Mean absolute jump across the join — 0 means an invisible seam."""
    view = _view(canvas, geo)
    kx0, ky0, kx1, ky1 = geo.keep_box
    keep_extent = (kx1 - kx0) if geo.axis == "x" else (ky1 - ky0)
    if keep_extent < 1 or keep_extent >= view.shape[1]:
        return 0.0
    step = np.abs(
        view[:, keep_extent - 1].astype(np.float32) - view[:, keep_extent].astype(np.float32)
    )
    return float(step.mean())


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def take_half(image: Image.Image, side: str) -> Image.Image:
    """Keep only ``side`` of a complete photo — for demos and for scoring the
    reconstruction against a ground truth we still hold."""
    side = (side or "").strip().lower()
    if side not in SIDES:
        raise ValueError(f"take must be one of {SIDES}.")
    width, height = image.size
    if side == "left":
        return image.crop((0, 0, width // 2, height))
    if side == "right":
        return image.crop((width - width // 2, 0, width, height))
    if side == "top":
        return image.crop((0, 0, width, height // 2))
    return image.crop((0, height - height // 2, width, height))


def complete(
    image: Image.Image,
    *,
    renderer: Renderer,
    side: str | None = None,
    gender: str = "",
    extra_prompt: str = "",
    seed: int = 7,
    tone_match: bool = True,
    seam_band: int | None = None,
    align_radius: int = 6,
    align_scale: float = 0.08,
    identity_anchor: bool = True,
    detector: Callable[[Image.Image], list[Any]] | None = None,
) -> dict[str, Any]:
    """Complete a half face. The supplied pixels come back untouched."""
    image = image.convert("RGB")
    geo, half, how = analyze(image, side=side, detector=detector)
    canvas = build_canvas(half, geo)
    keep = np.asarray(half.convert("RGB"))

    prompt = prompt_for(geo, gender=gender, extra=extra_prompt, anchor=identity_anchor)
    width, height = canvas.size
    # The bare half rides along as a second reference: the canvas tells the model
    # where everything goes, the half tells it whose face this is.
    references = [canvas, half] if identity_anchor else [canvas]
    render_image, meta = renderer(references, prompt, width, height)
    render = np.asarray(
        render_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    )

    render, transform, residual = align(
        render, keep, geo, radius=align_radius, scale_range=align_scale
    )

    gx0, gy0, gx1, gy1 = geo.gen_box
    kx0, ky0, kx1, ky1 = geo.keep_box
    generated = render[gy0:gy1, gx0:gx1]
    if tone_match:
        generated = _tone_match(generated, keep, geo)

    gen_extent = (gx1 - gx0) if geo.axis == "x" else (gy1 - gy0)
    band = seam_band if seam_band is not None else max(8, min(64, int(gen_extent * 0.05)))
    generated = feather_to_mirror(generated, keep, geo, band)

    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[gy0:gy1, gx0:gx1] = generated
    out[ky0:ky1, kx0:kx1] = keep          # last write wins: the supplied pixels

    if not np.array_equal(out[ky0:ky1, kx0:kx1], keep):
        raise RuntimeError(
            "Pixel-exactness check failed: the supplied half was modified. "
            "This is a bug — the result was discarded rather than returned."
        )

    return {
        "image": Image.fromarray(out),
        "canvas": canvas,
        "half": half,
        "geometry": geo,
        "pixel_exact": True,
        "side_detected_by": how,
        "align": transform,
        "align_residual": round(residual, 3),
        "seam_step": round(seam_step(out, geo), 3),
        "seam_band": band,
        "prompt": prompt,
        "render": meta,
    }


def verify(result: dict[str, Any], original: Image.Image) -> dict[str, Any]:
    """Score a completion against the ground-truth photo it was cut from."""
    out = np.asarray(result["image"].convert("RGB"), dtype=np.float32)
    ref = np.asarray(
        original.convert("RGB").resize(result["image"].size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    geo: Geometry = result["geometry"]
    gx0, gy0, gx1, gy1 = geo.gen_box
    gen_mae = float(np.abs(out[gy0:gy1, gx0:gx1] - ref[gy0:gy1, gx0:gx1]).mean())
    full_mae = float(np.abs(out - ref).mean())
    mse = float(np.mean((out[gy0:gy1, gx0:gx1] - ref[gy0:gy1, gx0:gx1]) ** 2))
    psnr = 99.0 if mse <= 1e-9 else float(10.0 * math.log10(255.0 * 255.0 / mse))
    return {
        "generated_half_mae": round(gen_mae, 2),
        "generated_half_psnr": round(psnr, 2),
        "full_image_mae": round(full_mae, 2),
    }
