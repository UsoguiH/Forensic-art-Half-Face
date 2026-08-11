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
    POOL_SEEDS,
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
    """Production pool shape, set by the v7 benchmark battery (2026-08-03).

    With a description: described arms + an idportrait control, ranked by the
    anchor — the battery split per subject (described forensic swept the CCTV
    subject, described forensicdoc swept the phone profile, the control led on
    the clean close-up), so the pool carries all three and the ranking picks.
    Without a description: described-skeleton forensic (pose+anti-idealization
    still help bare) + the proven idportrait fallback.
    """
    def slice_seeds(cursor: int, n: int) -> "tuple[int, ...]":
        # Disjoint per-arm seeds. The seed dominates the render, so two arms
        # sharing seed 7 produce near-identical images despite different
        # prompts — paid duplicates in the grid. Wraps past the pool with a
        # large offset so harness runs beyond 8 renders stay unique too.
        pool = len(POOL_SEEDS)
        return tuple(
            POOL_SEEDS[(cursor + j) % pool] + 10007 * ((cursor + j) // pool)
            for j in range(n)
        )

    if opts.styles:
        per_style = max(1, opts.candidates // len(opts.styles))
        return [
            Arm(style=s, seeds=slice_seeds(i * per_style, per_style))
            for i, s in enumerate(opts.styles)
        ]
    # Round-robin allocation so the pool is EXACTLY opts.candidates renders —
    # the old thirds/halves arithmetic under- or over-shot every count except
    # 6 and 8 (candidates=4 with a description rendered 3, candidates=1 bare
    # rendered 2). At 8 this reproduces the benchmark shape (3/3/2 and 4/4).
    order = (
        ("forensic", "forensicdoc", "idportrait")
        if opts.description
        else ("forensic", "idportrait")
    )
    counts = {style: 0 for style in order}
    for i in range(max(1, opts.candidates)):
        counts[order[i % len(order)]] += 1
    arms: list[Arm] = []
    cursor = 0
    for style, n in counts.items():
        if n:
            arms.append(Arm(style, slice_seeds(cursor, n)))
            cursor += n
    return arms


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
    # Auto-enrich on the evidence-poor tier ONLY — measured both ways: it
    # rescued weak CCTV bases (young8 0.41→0.51), and enabling it on the rich
    # tier was A/B'd on 2026-08-10 and REGRESSED the clean close-up (winner
    # 0.601→0.580, oracle 0.658→0.607): replace-outright swaps a strong
    # champion for a chain render. Do not widen this gate without re-measuring.
    do_enrich = opts.enrich if opts.enrich is not None else (ev.tier == "evidence-poor")
    if do_enrich and not mock and not champion.gated:
        champion, extra_cands, ledger = enrich_chain(
            ev, champion,
            renderer=renderer, detector=detector, embed=embed,
            description=opts.description, extra_prompt=opts.extra_prompt,
            out_size=opts.out_size, ledger=ledger,
        )
        candidates.extend(extra_cands)
        if ev.multi and ev.fused is not None:
            # Multi-photo: the fused anchor arbitrates across ALL rounds. The
            # blind replace-outright chain rule was measured on weak bases
            # (young8 0.41→0.51); with a description-strong base the chain
            # regressed the pick (0.4756→0.4614 vs truth) while the fused
            # anchor ranked the true best #1 — so let it pick globally.
            # Single-photo keeps replace-outright: min-both would wrongly keep
            # the base over a chain that improved identity (whatsapp
            # 0.606→0.634 measured exactly that).
            champion = pick(candidates, ev)
        enriched = champion.layout.startswith("enrich")

    return PipelineResult(
        winner=champion,
        candidates=candidates,
        tier=ev.tier,
        enriched=enriched,
        ledger=ledger,
        prompts_used=used_prompts,
    )
