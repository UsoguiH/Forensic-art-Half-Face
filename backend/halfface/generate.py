"""Render orchestration: candidate pools, the enrichment chain, spend ledger.

The renderer is a plain callable ``(references, prompt, width, height, seed) →
(image, meta)`` — fal today, the lab FLUX server tomorrow, a mock in tests.
This module never talks HTTP.

Enrichment (CCTV tier): the draft winner joins the references — never
edit-the-winner, which collapsed identity (0.28–0.36) every time it was tried.
Chain depth 2 with dedicated per-round seed tuples.  This replaces the v5
mechanism that sliced ``DEFAULT_SEEDS[4:6]`` out of a 4-seed tuple: both
slices were empty, the pool rendered nothing, and an assert 500'd every
evidence-poor upload after its 8 base renders were already paid for.

Every render appends to a Ledger; the estimated dollar figure is deliberately
conservative (fal FLUX.2 [dev] ≈ $0.03–0.05 per ~1 MP edit; we book 0.06).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from PIL import Image

from . import prompts as P
from .evidence import EvidenceSet
from .rank import Candidate, pick, score

Renderer = Callable[[list[Image.Image], str, int, int, int], "tuple[Image.Image, dict[str, Any]]"]

BASE_SEEDS: tuple[int, ...] = (7, 21, 77, 202)
# Seed pool for the base arms, sliced DISJOINTLY across arms: reusing
# BASE_SEEDS[:n] in every arm rendered the same seed under three near-identical
# prompts, and the seed dominates the render — the 8-card grid carried ~3
# unique takes and 5 paid duplicates. Values avoid ENRICH_ROUND_SEEDS.
POOL_SEEDS: tuple[int, ...] = (7, 21, 77, 202, 911, 1493, 2711, 3697)
ENRICH_ROUND_SEEDS: tuple[tuple[int, ...], ...] = ((42, 101), (301, 555))
EST_USD_PER_RENDER = 0.06
OUT_SIZE = (832, 1216)


@dataclass
class Arm:
    """One pool arm: a prompt style rendered at some seeds with a ref layout."""
    style: str
    seeds: Sequence[int]
    layout: str = "std"          # "std" | "grid"


@dataclass
class Ledger:
    renders: int = 0
    seconds: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, seconds: float) -> None:
        self.renders += 1
        self.seconds += seconds
        self.events.append({"label": label, "seconds": round(seconds, 2)})

    @property
    def est_usd(self) -> float:
        return round(self.renders * EST_USD_PER_RENDER, 2)

    def summary(self) -> dict[str, Any]:
        return {
            "renders": self.renders,
            "seconds": round(self.seconds, 1),
            "est_usd": self.est_usd,
            "est_usd_per_render": EST_USD_PER_RENDER,
        }


def _render_one(
    renderer: Renderer,
    refs: list[Image.Image],
    prompt: str,
    out_size: "tuple[int, int]",
    seed: int,
    label: str,
    ledger: Ledger,
) -> "tuple[Image.Image, dict[str, Any], float]":
    width, height = out_size
    t0 = time.time()
    image, meta = renderer(refs, prompt, width, height, seed)
    dt = time.time() - t0
    ledger.add(label, dt)
    image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, meta, dt


def render_pool(
    ev: EvidenceSet,
    arms: Sequence[Arm],
    *,
    renderer: Renderer,
    detector,
    embed,
    description: str = "",
    extra_prompt: str = "",
    out_size: "tuple[int, int]" = OUT_SIZE,
    ledger: "Ledger | None" = None,
) -> "tuple[list[Candidate], Ledger, dict[str, str]]":
    """Render every (style × seed) of every arm, scored and gated."""
    ledger = ledger or Ledger()
    used_prompts: dict[str, str] = {}
    out: list[Candidate] = []
    extra = (extra_prompt or "").strip()

    for arm in arms:
        refs = ev.refs_grid if (arm.layout == "grid" and ev.refs_grid) else ev.refs_std
        prompt = P.build(
            arm.style,
            who=ev.who,
            traits=ev.traits,
            description=description,
            n_photos=len(ev.photos),
            grid=(arm.layout == "grid" and ev.refs_grid is not None),
        )
        if extra:
            prompt = f"{prompt} {extra}"
        used_prompts[f"{arm.style}/{arm.layout}"] = prompt
        for seed in arm.seeds:
            image, meta, dt = _render_one(
                renderer, refs, prompt, out_size, seed,
                f"{arm.style}_{arm.layout}_{seed}", ledger,
            )
            cand = Candidate(
                image=image, style=arm.style, seed=seed, layout=arm.layout,
                render_meta=meta, seconds=dt,
            )
            out.append(score(cand, ev, detector, embed))
    return out, ledger, used_prompts


def enrich_chain(
    ev: EvidenceSet,
    champion: Candidate,
    *,
    renderer: Renderer,
    detector,
    embed,
    description: str = "",
    extra_prompt: str = "",
    out_size: "tuple[int, int]" = OUT_SIZE,
    ledger: "Ledger | None" = None,
    rounds: "tuple[tuple[int, ...], ...]" = ENRICH_ROUND_SEEDS,
) -> "tuple[Candidate, list[Candidate], Ledger]":
    """Draft-as-reference enrichment, chain depth = len(rounds).

    Each round renders the champion's own style with the draft appended to the
    references.  The round's gated pick REPLACES the champion outright (never
    score-compared across rounds: enrichment pulls renders away from the
    profile pose, so scores sit lower even when identity improved).  A round
    whose pick is gated breaks the chain.
    """
    ledger = ledger or Ledger()
    extras: list[Candidate] = []
    base_refs = ev.refs_std[:3]
    extra = (extra_prompt or "").strip()

    for i, seeds in enumerate(rounds, start=1):
        layout = f"enrich{i}"
        prompt = P.build(
            champion.style if champion.style != "idportrait" else "forensic",
            who=ev.who,
            traits=ev.traits,
            description=description,
            n_photos=len(ev.photos),
            grid=False,
        )
        prompt = f"{prompt} {P.ENRICH_SENTENCE}"
        if extra:
            prompt = f"{prompt} {extra}"
        refs = base_refs + [champion.image]
        round_cands: list[Candidate] = []
        for seed in seeds:
            image, meta, dt = _render_one(
                renderer, refs, prompt, out_size, seed, f"{layout}_{seed}", ledger,
            )
            cand = Candidate(
                image=image, style=champion.style, seed=seed, layout=layout,
                render_meta=meta, seconds=dt,
            )
            round_cands.append(score(cand, ev, detector, embed))
        extras.extend(round_cands)
        round_pick = pick(round_cands, ev)
        if round_pick is None or round_pick.gated:
            break
        champion = round_pick
    return champion, extras, ledger
