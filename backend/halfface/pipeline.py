"""Orchestrator: EvidenceSet + options → winner and full candidate pool.

Free of HTTP/UI concerns.  ``backend.py`` adapts HalfFaceReq to Options and
PipelineResult back to the existing response shape; harnesses call this
directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .evidence import EvidenceSet
from .generate import (
    BASE_SEEDS,
    OUT_SIZE,
    Arm,
    Ledger,
    Renderer,
    enrich_chain,
    render_pool,
)
from .rank import Candidate, pick


@dataclass
class Options:
    candidates: int = 8                       # base pool size target
    styles: "tuple[str, ...] | None" = None   # None = auto by evidence
    arms: "Sequence[Arm] | None" = None       # explicit pool spec wins over styles
    description: str = ""                     # examiner identity notes (v7 signal)
    extra_prompt: str = ""                    # caller free-text, appended last
    out_size: "tuple[int, int]" = OUT_SIZE
    enrich: "bool | None" = None              # None = auto (evidence-poor tier only)


@dataclass
class PipelineResult:
    winner: Candidate
    candidates: list[Candidate] = field(default_factory=list)
    tier: str = ""
    enriched: bool = False
    ledger: "Ledger | None" = None
    prompts_used: dict = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "enriched": self.enriched,
            "winner": self.winner.meta() if self.winner else None,
            "candidates": [c.meta() for c in self.candidates],
            "ledger": self.ledger.summary() if self.ledger else None,
        }


def default_arms(ev: EvidenceSet, opts: Options) -> list[Arm]:
    """Two-style pool, seeds split evenly — the shape every battery used.

    With an examiner description the pool is (forensic, forensicdoc); without
    one, forensic still carries pose + anti-idealization and idportrait is the
    proven generic fallback.
    """
    styles = opts.styles or (
        ("forensic", "forensicdoc") if opts.description else ("forensic", "idportrait")
    )
    per_style = max(1, opts.candidates // max(1, len(styles)))
    return [Arm(style=s, seeds=BASE_SEEDS[:per_style]) for s in styles]


def reconstruct_frontal(
    ev: EvidenceSet,
    *,
    renderer: Renderer,
    detector,
    embed,
    opts: "Options | None" = None,
) -> PipelineResult:
    opts = opts or Options()
    mock = embed is None or detector is None

    arms = list(opts.arms) if opts.arms else default_arms(ev, opts)
    if mock:  # keep mock runs to a single render, like the legacy engine
        arms = [Arm(style=arms[0].style, seeds=tuple(arms[0].seeds)[:1], layout=arms[0].layout)]

    ledger = Ledger()
    candidates, ledger, used_prompts = render_pool(
        ev, arms,
        renderer=renderer, detector=detector, embed=embed,
        description=opts.description, extra_prompt=opts.extra_prompt,
        out_size=opts.out_size, ledger=ledger,
    )
    champion = pick(candidates, ev)
    if champion is None:
        raise RuntimeError("The render pool produced no candidates.")

    enriched = False
    do_enrich = opts.enrich if opts.enrich is not None else (ev.tier == "evidence-poor")
    if do_enrich and not mock and not champion.gated:
        champion, extra_cands, ledger = enrich_chain(
            ev, champion,
            renderer=renderer, detector=detector, embed=embed,
            description=opts.description, extra_prompt=opts.extra_prompt,
            out_size=opts.out_size, ledger=ledger,
        )
        candidates.extend(extra_cands)
        enriched = champion.layout.startswith("enrich")

    return PipelineResult(
        winner=champion,
        candidates=candidates,
        tier=ev.tier,
        enriched=enriched,
        ledger=ledger,
        prompts_used=used_prompts,
    )
