"""halfface — v7 "Witness" profile→frontal reconstruction package.

Full redesign of the v3/v4/v5 engine (faceid_reconstruct2.py) around one
measured finding: FLUX idealizes every attribute the prompt does not pin down
(facial hair densified, hairlines restored, faces narrowed, brows bolded, skin
smoothed).  v7 injects a concrete per-subject *examiner description* plus an
anti-idealization directive into a pose-first prompt, keeps every mechanism the
earlier versions measured as a win (head-shoulders crop, tight-crop extras,
mirror ref, fused quality-weighted anchor, frontality/suspicious gates,
CCTV-tier enrichment chain), and fixes the enrichment seed bug that crashed
every evidence-poor upload (backend.py:1375 sliced an empty seed tuple).

Open-source models only: FLUX.2 [dev] (fal-hosted or lab server) renders;
InsightFace buffalo_l embeds and ranks locally.  No post-processing models.
"""
from .pipeline import Options, PipelineResult, reconstruct_frontal
from .evidence import EvidenceSet, build_evidence

VERSION = "v7.0"

__all__ = [
    "VERSION",
    "Options",
    "PipelineResult",
    "reconstruct_frontal",
    "EvidenceSet",
    "build_evidence",
]
