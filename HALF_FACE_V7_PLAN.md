# HALF FACE v7 — "Witness": generation-side redesign (plan of record, 2026-08-03)

Goal: from a side-profile / CCTV photo (+ optional extra photos of the same person), produce a frontal
portrait the user judges as *the same person*. Full refactor of the v3/v4/v5 engine. **Open-source
models only** (FLUX.2 [dev] via fal edit endpoint; local InsightFace buffalo_l for ranking). **No
post-processing models** (no swap, no restoration — banned by user directive and by the forensic
literature already cited in HALF_FACE_V3_PLAN.md). **fal budget ≈ $5 total: build first, then one
frugal benchmark pass (~30–36 renders), no wide sweeps.**

## 1. Diagnosis — why current results look wrong

Visual audit of the v5 champions vs targets/truth (all four subjects) shows one systematic failure:
**FLUX idealizes what the prompt does not pin down.** Concretely, on every subject:

1. Facial hair is amplified: sparse/patchy mustache or jaw scruff → dense groomed mustache/beard
   (young8, nawaf, man2).
2. Hairline is regularized: nawaf's clear temple recession → full juvenile hairline.
3. Face is re-proportioned toward the "attractive oval": young8's short broad face → long narrow face.
4. Brows are bolded and straightened; eyes enlarged and rounded; skin smoothed and paled.

The v5 prompts describe the *task* ("same person, ID portrait, no beautification") but never the
*person*. The negative directives ("no beautification") lose to the model prior because nothing
states what is actually there. Meanwhile idportrait2 (generic axis-enumeration: "keep the exact head
shape, face width, …") was measured WORSE — directives without content push outputs generic.

**Conclusion: the missing signal is a concrete, per-subject, observed attribute description.**
Everything measurable from a profile is pose-invariant in the vertical axis (hairline height, brow–eye
gap, nose length, philtrum, lip thickness, chin height, forehead slope, nose shape) and facial-hair
density/pattern + hairline state are directly observable. That content never reached the model in
v1–v5.

Also found (code audit): **production bug** — backend.py:1375 slices enrichment seeds
`f2.DEFAULT_SEEDS[4:6]`/`[6:8]` but `faceid_reconstruct2.DEFAULT_SEEDS` has only 4 entries → both
slices empty → `assert best is not None` crashes → **every evidence-poor (CCTV, face<200px) upload
500s after paying for the 8 base renders**. v7 fixes this.

## 2. What we keep (measured, still true)

- Routing (pose-first classify; 31/31 battery) — unchanged.
- head_shoulders_crop 3:4 as primary ref; extra photos as TIGHT face crops (≤3); mirror ref;
  832×1216 output; LANCZOS-only normalization (no generative SR).
- Pool → gates (frontality ≥ 0.55, suspicious ≥ 0.74) → fused/min-both ranking; human-overrule
  candidate list to UI.
- Fused quality-weighted multi-photo anchor (picked the oracle on every multi-photo run).
- Enrichment only on CCTV tier, draft-as-reference (never edit-the-winner), chain depth 2.
- fal_provider (fit_size, queue fallback, klein fallback, spend meta).

## 3. v7 architecture (new package `backend/halfface/`)

```
backend/halfface/
  __init__.py      — public API: reconstruct_frontal(...),版本 tag
  evidence.py      — intake: decode, detect, classify tier, crops (hs/tight/mirror), quality
                     weights (det_score·size·sharpness), fused anchor, EvidenceSet dataclass
  describe.py      — per-subject forensic description:
                       (a) measured: vertical proportions + yaw + face-size tier from landmarks
                       (b) observed: analyst/VLM-provided attribute lines (pluggable; file or field)
                       (c) assembler → compact "examiner description" block (≤ ~90 words)
  prompts.py       — v7 prompt system (see §4): build(style, who, traits, description, n_refs)
  rank.py          — gates + ranking + honest metrics (LOO-fused, vs-truth when supplied)
  generate.py      — render pool orchestration over fal_provider/ENGINE renderer (provider-agnostic
                     callable), enrichment chain (dedicated ENRICH_SEEDS, bug-proof), spend ledger
  pipeline.py      — reconstruct_frontal(EvidenceSet, opts) → PipelineResult
                     (winner, candidates+meta, tier, ledger) — no HTTP/UI concerns
```
`backend.py::_frontalize_profile` becomes a thin adapter: HalfFaceReq → EvidenceSet → pipeline →
existing response shape (API unchanged, UI untouched). `faceid_reconstruct2.py` stays on disk for
reference; nothing imports it after the rewire.

## 4. Prompt system (the core change)

Skeleton = v5 "forensic" pose-first framing (the only prompt that ever won a subject on merit),
now with three blocks:

1. **Pose block** (first, unconditional): direct-to-camera ID portrait, head level, eyes to lens,
   plain backdrop, same clothing.
2. **Identity block** (NEW): the per-subject examiner description — only *observed concrete facts*,
   never generic axis names. Example (young8):
   "Short, broad face with full cheeks and a soft jawline; dark wavy voluminous hair; thin sparse
   mustache, faint scruff on the chin only; medium-thick softly arched brows; slightly hooded,
   medium-size dark eyes; prominent straight nose with a rounded tip; full lower lip; rounded,
   slightly recessed chin; warm light-brown skin with natural texture."
3. **Anti-idealization block** (NEW, short): "This is an ordinary unretouched person, not a model:
   do not beautify, do not thicken or groom the facial hair beyond the photos, do not restore the
   hairline, keep natural asymmetries and skin texture."

Two styles ship: `forensic` (studio/ID framing) and `forensicdoc` (documentary/scene-neutral) — both
carry all three blocks; multi-photo adds the "every photo shows the same person; trust only features
the photos agree on" preamble from multiref. A plain `idportrait` control seed is kept in benchmark
pools as regression insurance.

Description sourcing now: measured block computed from landmarks (free, deterministic); observed
block authored per subject from the evidence (analyst role — Claude in-session; later a local
Qwen2.5-VL on the lab GPU; also exposed as `HalfFaceReq.description` for a human examiner).

## 5. Reference layouts (second lever, pooled not swept)

- `refs-std` (v5 winner): [hs_crop, tight₁, tight₂, tight₃] or [hs_crop, mirror] single-photo.
- `refs-grid` (NEW): single composite 2-up/4-up "identity sheet" (profile | mirror / tights) as
  image 1 + hs_crop as image 2 — asserts co-identity structurally, frees ref slots. Included as a
  pool arm on one subject only (cheapest honest A/B); adopted wider only if it wins there.

## 6. Benchmark protocol (honest metrics, previous bests to beat)

New fact unlocked by recon: the young8 CCTV cluster and the whatsapp pair are THE SAME PERSON —
young8 gains a REAL ground truth (WhatsApp ...6.38.17 AM (1).jpeg). AI-panel targets are kept only
as legacy columns (their own agreement with evidence caps at 0.366–0.527 — matching them harder
means matching hallucinations).

| subject  | evidence                                   | honest primary metric            | previous best (v5 champions) |
|----------|--------------------------------------------|----------------------------------|------------------------------|
| whatsapp | clean profile (1 photo)                    | ArcFace vs REAL truth            | 0.6339 |
| young8   | in_young8_cctv + clean_profile + near_cctv + in_young7_cctv | vs REAL truth (new) + fused-evidence | fused 0.5207 (vs AI target 0.5333) |
| man2     | in_man2/man1/man3_cctv                     | fused-evidence + LOO             | fused 0.4711 (vs AI target 0.4849) |
| nawaf    | nawaf.jpeg (1 photo)                       | min_both + visual                | 0.5345 |

Success = beat every previous-best primary metric with the SAME pool budget the champions used
(8 base renders), plus visual verdict. Post-run, champions + boards land in Test_AI_Half/Final_Results.

## 7. Spend plan (hard ceiling ≈ $2.5 of the $5)

- 0 renders: full local dry-run with a mock renderer (unit-level: crops, prompts, gates, ledger).
- 1 render: wiring smoke (whatsapp, 1 seed) — also confirms balance is live.
- Benchmark pools: whatsapp 8, young8 8, man2 8, nawaf 6 (single-photo, no grid arm).
- Enrichment (CCTV tier only, chain≤2, 2 renders/round): man2, young8 if base pick survives gates.
- Contingency: ≤ 6 targeted renders for ONE failing subject, once. Nothing else.
Every render logged to a ledger (count × est. $/render) printed at each run's end.

## 8. Wiring & hygiene

- Fix enrichment seed bug via dedicated `ENRICH_SEEDS=(42, 101, 301, 555)`.
- Keep HalfFaceReq schema; add optional `description: str|None` (≤600 chars) — examiner text.
- tools/iterate_v7.py — same CLI spirit as v3 harness; subjects defined in a small registry with
  paths + descriptions; `--mock` for zero-cost runs.
- Commit on Feature/Half_Face; push to UsoguiH/Forensic-art-Half-Face; update memory.

## 9. Execution order (tasks)

1. Plan doc (this file) ✅
2. Package `backend/halfface/` (evidence, describe, prompts, rank, generate, pipeline)
3. Harness tools/iterate_v7.py + subject registry incl. per-subject descriptions
4. Mock dry-run battery (0 renders) — correctness + prompt review
5. Smoke render (1) then benchmark pools (~30) with decision rules from §7
6. Rewire backend._frontalize_profile → halfface package (+bug fix), routing battery offline re-run
7. Final_Results refresh (winners + boards + report.json), commit + push, memory update
