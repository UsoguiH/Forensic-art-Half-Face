"""Gates, ranking and honest metrics for rendered candidates.

Rules carried over because they were measured, not because they are pretty:

- Frontality gate ≥ 0.55 — the one failure identity ranking cannot see is a
  render that kept the head turned: it looks exactly like fidelity.
- Suspicious gate — two real cross-pose photos of one person land ~0.55–0.70;
  a candidate scoring > 0.74 against the *source* embedding copied the pose or
  the numbers.  Single-photo: min_both > 0.74.  Multi-photo: max per-photo
  sim > 0.74 (it cloned ONE photo instead of triangulating).
- Rank score — multi-photo: cosine to the fused quality-weighted anchor
  (picked the pool oracle on every measured multi-photo run); single-photo:
  min(sim-to-profile, sim-to-mirror) ("must resemble both sides", lifted the
  worst pick 0.445 → 0.513).
- If EVERY candidate is gated the whole pool becomes eligible again — a gated
  best beats an empty response, and the UI ships all candidates for human
  overrule anyway (rank-1 vs rank-3 is a coin flip at CCTV anchor quality).

Honest evaluation (report-only, never used to pick): vs_truth against a REAL
frontal when one exists.  AI-panel targets are legacy columns — their own
agreement with the evidence caps at 0.37–0.53, so beyond that they score the
target-generator's hallucinations, not the person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

import face_crop

from .evidence import EvidenceSet, cos, unit

MIN_FRONTALITY = 0.55
SUSPICIOUS_SIMILARITY = 0.74


@dataclass
class Candidate:
    image: Image.Image
    style: str
    seed: int
    layout: str = "std"                      # std | grid | enrich1 | enrich2
    frontality: float = 0.0
    min_both: "float | None" = None
    sim_fused: "float | None" = None
    per_photo: list = field(default_factory=list)
    suspicious: bool = False
    gated: bool = False
    gate_reason: str = ""
    render_meta: dict = field(default_factory=dict)
    seconds: float = 0.0

    def meta(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "seed": self.seed,
            "layout": self.layout,
            "frontality": self.frontality,
            "min_both": _r(self.min_both),
            "sim_fused": _r(self.sim_fused),
            "per_photo": [_r(v) for v in self.per_photo],
            "suspicious": self.suspicious,
            "gated": self.gated,
            "gate_reason": self.gate_reason,
            "seconds": round(self.seconds, 2),
        }


def _r(v: "float | None") -> "float | None":
    return None if v is None else round(float(v), 4)


def score(cand: Candidate, ev: EvidenceSet, detector, embed) -> Candidate:
    """Fill metrics + gates in place, return the candidate."""
    face = face_crop.biggest_face(detector(cand.image)) if detector else None
    cand.frontality = face_crop.frontality(face)
    v = unit(embed(cand.image)) if embed else None
    if v is not None:
        cand.min_both = None
        if ev.v_primary is not None and ev.v_mirror is not None:
            cand.min_both = min(cos(v, ev.v_primary), cos(v, ev.v_mirror))
        cand.sim_fused = cos(v, ev.fused)
        cand.per_photo = ev.per_photo_sims(v)
        if ev.multi:
            peak = max((s for s in cand.per_photo if s is not None), default=None)
            cand.suspicious = bool(peak is not None and peak > SUSPICIOUS_SIMILARITY)
        else:
            cand.suspicious = bool(
                cand.min_both is not None and cand.min_both > SUSPICIOUS_SIMILARITY
            )
    reasons = []
    if cand.frontality < MIN_FRONTALITY:
        reasons.append(f"frontality {cand.frontality:.2f} < {MIN_FRONTALITY}")
    if cand.suspicious:
        reasons.append("suspicious (source clone)")
    cand.gated = bool(reasons)
    cand.gate_reason = "; ".join(reasons)
    return cand


def rank_key(cand: Candidate, ev: EvidenceSet) -> float:
    if ev.multi and cand.sim_fused is not None:
        return cand.sim_fused
    if cand.min_both is not None:
        return cand.min_both
    return -1.0


def pick(candidates: "list[Candidate]", ev: EvidenceSet) -> "Candidate | None":
    if not candidates:
        return None
    pool = [c for c in candidates if not c.gated] or candidates
    return max(pool, key=lambda c: rank_key(c, ev))


def vs_image(cand_image: Image.Image, ref_image: "Image.Image | None", embed) -> "float | None":
    """Report-only cosine of a candidate against a truth/target photo."""
    if ref_image is None or embed is None:
        return None
    v = unit(embed(cand_image))
    t = unit(embed(ref_image))
    return None if (v is None or t is None) else round(float(cos(v, t)), 4)
