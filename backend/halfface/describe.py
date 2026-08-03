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
