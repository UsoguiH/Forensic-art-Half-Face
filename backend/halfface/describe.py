"""Per-subject forensic description — the v7 identity signal.

The visual audit of every v3–v5 champion found one systematic failure: FLUX
idealizes whatever the prompt leaves open (sparse mustache → groomed mustache,
receding hairline → full hairline, short broad face → long narrow face, brows
bolded, eyes enlarged, skin smoothed).  Generic directives do not fix this —
the axis-enumeration prompt (idportrait2) measured WORSE, because naming axes
without content pushes outputs generic.  What works is stating the observed
facts themselves.

A description has two sources:

- ``measured_lines``  deterministic, from the detector: sex, coarse age (only
  when the face is big enough to trust), facing side.  Deliberately minimal —
  5-point landmarks on a 90° profile cannot support proportion claims, and a
  wrong "measurement" in the prompt is worse than none.
- ``observed``        free text from an examiner looking at the evidence:
  today an analyst (or Claude) writes it; on the lab box a local open VLM
  (Qwen2.5-VL) can fill the same field; the API exposes it as
  ``HalfFaceReq.description``.

``compose`` folds both into one compact block for prompts.build().  Hard cap
~600 chars (sentence-boundary truncation): FLUX follows short concrete lists;
walls of text dilute.
"""
from __future__ import annotations

from typing import Any

MAX_CHARS = 600


def measured_lines(face: Any, face_px: int) -> list[str]:
    lines: list[str] = []
    if face is None:
        return lines
    try:
        import numpy as np

        kps = np.asarray(face.kps, dtype=np.float32)
        x0 = float(face.bbox[0]); x1 = float(face.bbox[2])
        nose_x = float(kps[2][0])
        mid = (x0 + x1) / 2.0
        off = (nose_x - mid) / max(1.0, x1 - x0)
        if abs(off) > 0.18:
            side = "left" if off < 0 else "right"
            lines.append(f"photographed in profile facing {side}")
    except Exception:
        pass
    return lines


def compose(observed: str = "", measured: "list[str] | None" = None) -> str:
    """One examiner-description block, ≤ MAX_CHARS, sentence-safe."""
    parts: list[str] = []
    obs = (observed or "").strip().rstrip(".")
    if obs:
        parts.append(obs + ".")
    for m in measured or []:
        m = m.strip().rstrip(".")
        if m:
            parts.append(m[0].upper() + m[1:] + ".")
    text = " ".join(parts)
    if len(text) <= MAX_CHARS:
        return text
    cut = text[:MAX_CHARS]
    dot = cut.rfind(".")
    return (cut[: dot + 1] if dot > 40 else cut).strip()


# ---------------------------------------------------------------------------
# VLM auto-description (Qwen3-VL via backend/vlm.py)
# ---------------------------------------------------------------------------
# The extraction prompt encodes every measured v7/v8 rule: observed facts only
# (axis-enumeration measured WORSE), explicit facial-hair negatives, clothing/
# headwear stated (unstated clothing made every model invent a shirt and tie
# on a thobe-wearing subject), no hedging about what is hidden.

VLM_PROMPT_FULL = (
    "You are a forensic examiner writing identity notes about the ONE person shown in "
    "the photo(s). The notes will guide an artist drawing this exact person from the "
    "front, so every wrong word damages the likeness.\n"
    "Rules:\n"
    "- Describe ONLY what is clearly visible. If a feature is not clearly visible, do "
    "not mention it at all. Never guess, never hedge (no 'probably', 'seems', "
    "'appears', 'likely', 'possibly', 'hidden', 'not visible').\n"
    "- This is an ordinary real person: record non-ideal features honestly (patchy or "
    "sparse facial hair, receding hairline, tired eyes, uneven skin) — do not "
    "beautify.\n"
    "- Facial hair needs explicit negatives where true: e.g. 'a thin mustache only — "
    "no beard, the cheeks and chin are clean-shaven'.\n"
    "- End with the clothing and any headwear exactly as worn (use the correct terms: "
    "thobe, kufi prayer cap, shemagh, t-shirt...).\n"
    "- Do NOT mention: pose, camera, background, image quality, or that this is a "
    "photo.\n"
    "- Format: plain prose noun-phrases in this order — approximate age; hair (volume, "
    "curl, hairline); face shape; nose; facial hair; eyebrows; eyes; lips; skin tone "
    "and texture; clothing/headwear. One compact paragraph, under 480 characters, no "
    "list, no markdown, no trailing period.\n"
    "If several photos are given they show the same person: describe only features "
    "the photos agree on.\n"
    "If the photo is blurry, low-resolution or surveillance footage: state only the "
    "features you are CERTAIN of and simply leave the rest out — especially face "
    "shape, skin texture, exact age and facial-hair density, which blur distorts. "
    "A short list of certain facts is worth more than full coverage with one wrong "
    "guess: one wrong geometric claim ruins the likeness."
)

VLM_PROMPT_LITE = (
    "You are a forensic examiner writing minimal identity notes about the ONE person "
    "shown. Output a single short phrase, under 150 characters, no markdown, no "
    "trailing period, covering ONLY: approximate age; hair; facial hair with explicit "
    "negatives where true (e.g. 'no beard, cheeks clean-shaven'); headwear if any. "
    "Only what is clearly visible — never guess, never hedge, do not beautify, no "
    "pose/camera/background words."
)

_HEDGES = (
    "probably", "seems", "appears", "likely", "possibly", "hidden", "not visible",
    "cannot", "can't", "unclear", "unable", "hard to", "difficult to", "may be",
    "might", "perhaps", "obscured", "i'm sorry", "i cannot", "as an ai",
    "the photo", "the image", "the picture", "profile", "camera", "background",
)


def _scrub(text: str) -> str:
    """Deterministic cleanup of VLM output: strip markup, drop hedged/banned
    CLAUSES, normalize whitespace. Clause-level, not sentence-level — the model
    tends to emit one long comma-run sentence, and one stray banned word must
    not wipe the whole description. Returns '' when nothing trustworthy is left."""
    import re

    text = (text or "").strip()
    for token in ("```", "**", "##", "“", "”", '"'):
        text = text.replace(token, "")
    text = " ".join(text.split())
    kept: list[str] = []
    for clause in re.split(r"[.;,]\s+", text):
        c = clause.strip().rstrip(".,;")
        if c and not any(h in c.lower() for h in _HEDGES):
            kept.append(c)
    out = ", ".join(kept).strip()
    return out if len(out) >= 80 else ""


def from_image(ev: Any, *, extras_limit: int = 2) -> "tuple[str, dict]":
    """Auto-write the examiner description from the evidence photos.

    Returns (description, meta). description == "" means "proceed exactly as an
    empty analyst field does today" — the caller never blocks on the VLM.
    Tier policy is measured, not configurable: full notes on the evidence-poor
    CCTV tier; LITE notes on clean close-ups (v7: full descriptions LOSE to
    pixel-trusting prompts there — text competes with strong pixels).
    """
    import time as _time

    meta: dict[str, Any] = {"source": "none"}
    try:
        import vlm
    except ImportError:  # tools/ path setups
        try:
            from backend import vlm  # type: ignore[no-redef]
        except ImportError:
            return "", meta
    if not vlm.available():
        return "", meta

    # Always the FULL examiner prompt. v7's production answer to "descriptions
    # lose on clean close-ups" is the POOL (idportrait control arm + anchor
    # ranking decides per subject), not thinner text: tight profile crops
    # classify evidence-rich yet measurably win WITH their full descriptions
    # (beard/pair4 in v8). LITE remains available for future experiments.
    lite = False
    prompt = VLM_PROMPT_FULL
    images = [ev.working] + [p.tight for p in ev.photos[1:1 + extras_limit]]

    t0 = _time.time()
    try:
        raw = vlm.chat_vision(prompt, images, max_tokens=400)
    except Exception as exc:
        print(f"[describe] VLM failed, rendering without description: {exc}", flush=True)
        meta["error"] = str(exc)[:200]
        return "", meta

    observed = _scrub(raw)
    if not observed:
        meta.update({"source": "none", "raw_chars": len(raw)})
        return "", meta
    if lite and len(observed) > 200:
        observed = observed[:200].rsplit(" ", 1)[0]

    primary = ev.photos[0] if getattr(ev, "photos", None) else None
    measured = measured_lines(getattr(primary, "face", None),
                              getattr(primary, "face_px", 0)) if primary else []
    text = compose(observed=observed, measured=measured)
    meta.update({
        "source": "vlm", "model": vlm.model(), "lite": lite,
        "chars": len(text), "latency_s": round(_time.time() - t0, 1),
    })
    return text, meta
