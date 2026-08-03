# Half-Face v3 — "Any Photo → Identity-Faithful Frontal ID Portrait"

Plan of record, 2026-08-03. Input domain: anything a user uploads — far CCTV frames
(40–160 px faces), phone profile photos (~250 px), WhatsApp close-ups, genuine
half-of-frontal crops. Output: photorealistic frontal head-and-shoulders ID
portrait of the same person + honest identity confidence. Renderer: fal.ai for
now; every model chosen must be open-weights so the whole stack can move to the
offline lab GPU (~24 GB) later. Research basis: three deep literature/ecosystem
sweeps (editors · frontalization methods · enhancement+recognition), 2026-08-03.

---

## 1. Research verdicts that fix the architecture

1. **Pixel-referenced diffusion editing is the correct and only competitive
   route.** Every alternative fails structurally at 90° yaw: 3DMM routes
   (DECA/EMOCA/MICA/3DDFA_V3) fill the hidden half with generic PCA texture;
   StyleGAN inversion and reenactment (LivePortrait etc.) require near-frontal
   input; GAN-era frontalizers (TP-GAN, Rotate-and-Render) are frozen 2018-2020
   artifacts. 2026 community practice for "turn head to camera, keep identity"
   is FLUX.2 / Qwen-Image-Edit multi-reference editing — the same family we
   already measured best (0.665 vs truth; real profile scores 0.630).
2. **Embedding-locked generators stay banned** (PuLID, InstantID, InfiniteYou,
   Arc2Face, IP-Adapter-FaceID, DreamO-ID): they decode the pose-biased ArcFace
   vector — our own measurements (0.46–0.60 vs truth) are corroborated by the
   literature (ArcFace profile error >10×; DREAM exists precisely because the
   bias needs learned correction). Arc2Face's pose-ControlNet controls rendered
   pose, not the vector's contamination — doesn't help.
3. **Face-restoration models are forensically dangerous.** Peer-reviewed
   evidence: CodeFormer degraded forensic likelihood-ratio reliability
   (PMC10938076) and at 8× downsampling produced *worse* recognition than doing
   nothing (Norman & Farid CVPR'24W); GFPGAN has author-admitted identity
   drift; SUPIR/StableSR have admitted hallucination/small-face failures.
   → identity embeddings are ALWAYS computed on raw crops; visual reference
   enhancement uses only priorless SR (Real-ESRGAN x4plus, BSD-3, no
   --face_enhance) + a mandatory NTIRE-style cosine gate (enhanced vs raw,
   reject below ~0.4).
4. **Multi-frame fusion is the biggest evidenced identity lever.**
   Filter-then-quality-weight template fusion (TransFIRA-style) gains
   +8 to +62 pp TAR at strict FAR on surveillance benchmarks vs naive
   averaging. When the user has video/several frames, this beats any
   single-image cleverness. Naive per-frame quality weighting alone is NOT
   reliable — filter first, then weight.
5. **AdaFace (IR101, WebFace12M; MIT) is the one worthwhile embedder upgrade** —
   TinyFace rank-1 ~72–74 % vs ArcFace 62–71 % on tiny faces; parity (not
   better) cross-pose. buffalo_l stays as detector; AdaFace joins for ranking
   on small faces. (antelopev2: no evidence of gain — skip. TransFace: no
   consumable weights — skip.)
6. **Licenses matter and are currently dirty.** FLUX.2 [dev]/[klein-9B] are
   non-commercial (fal.ai usage is covered by fal's hosting terms for now;
   offline operational deployment needs a BFL commercial license or a swap);
   InsightFace weights are research-only; CodeFormer/KEEP/StableSR are NTU
   S-Lab non-commercial. Fully-open alternates exist for every slot:
   **Qwen-Image-Edit-2509 (Apache 2.0)** for the editor, **AdaFace (MIT)** for
   the embedder, **Real-ESRGAN (BSD-3)** for SR, 3DDFA_V3 (MIT) for geometry.
7. **Physics ceiling**: at true 90° yaw an entire hemi-face is absent, not
   degraded — the generated far side is plausible, never verified. Report
   confidence honestly; never imply the hidden half is evidence.

## 2. v3 architecture

```
upload(s: 1..N images or clip)
   │
   ▼
[0 ROUTE]     pose-first classify (keypoints decide; 2% area floor only when
              no keypoints). half → pixel-exact mirror pipeline (unchanged).
              frontal → pass-through/portrait re-stage. profile → v3 path:
   ▼
[1 NORMALIZE] detect → (multi-frame: FIQA triage, SER-FIQ/CR-FIQA-S, drop
              junk frames) → head-and-shoulders crop 3:4 (~2.2× face width,
              headwear margin, avoid timestamp bands) → re-detect in crop
   ▼
[2 EVIDENCE]  two strictly separated paths:
              · VISUAL: face <300 px → Real-ESRGAN x4plus (no face prior);
                cosine gate vs raw crop (~0.4) or fall back to raw
              · IDENTITY: embeddings from RAW crops only;
                multi-frame → filter-then-quality-weight fusion;
                single frame → profile+mirror fused vector (status quo)
   ▼
[3 GENERATE]  FLUX.2 edit (fal now / lab later), refs [crop, mirror(crop)]
              (+ up to 2 extra distinct frames), FIXED portrait size
              ~832×1216, pool = 4 seeds × 2 prompt styles:
              "descriptive-studio" (measured winner) + new "id-portrait"
              (studio beige backdrop, same clothing/headwear, head raised,
              eyes to lens, low-res-surveillance-aware). Auto-inject
              detected age/sex.
   ▼
[4 SELECT]    gates: frontality (kps), det_score, sharpness, attribute
              consistency (candidate age within ~12y of source, same sex —
              FLUX age-drift is documented, FLUXSynID) → rank by
              min(sim-to-profile, sim-to-mirror) PLUS a consensus term
              (mean similarity to the other candidates: hallucinations
              diverge, real features recur — and consensus does not depend
              on the noisy tiny-crop reference embedding; mixing weight is
              an ablation) [AdaFace on small faces, buffalo_l otherwise]
              → SUSPICIOUS_SIMILARITY guard stays
   ▼
[5 REPORT]    winner + full candidate grid + confidence tier
              (evidence-rich ≥250 px vs evidence-poor CCTV) + honest caveat
```

## 3. What changes vs v2 (faceid_reconstruct2)

| # | Change | Why |
|---|--------|-----|
| 1 | classify(): pose before area floor | 7/16 New DataSet misroute to mirror pipeline today |
| 2 | Canonical head-and-shoulders crop for every input | FLUX attention + output resolution spent on the face, not the corridor |
| 3 | Fixed portrait render size | stop rendering at scene size |
| 4 | "id-portrait" prompt replaces "preserving" on the profile route | preserving keeps the corridor; targets the user's reference output look |
| 5 | Conservative SR + cosine gate for tiny faces (visual path only) | evidence: face-restorers hallucinate; priorless SR is the safe option |
| 6 | Embeddings always from raw crops | forensic literature; enhancement never touches the identity path |
| 7 | Multi-frame input (N images/clip) + FIQA triage + filter-then-weight fusion | biggest evidenced identity gain |
| 8 | AdaFace ranking embedder for small faces | TinyFace +10 pp over ArcFace class |
| 9 | Age/sex auto-injection into prompts | free from buffalo_l genderage |
| 10 | Confidence tiers in response | 90°/tiny-face ceiling must be visible to the user |

Not doing (measured/evidenced dead ends): refinement edit loop (identity
collapse 0.28–0.36, twice), embedding-locked generators, CodeFormer/GFPGAN/
SUPIR anywhere, antelopev2, reenactment/3DMM as the frontalizer.

## 4. License / deployment matrix

| Component | Now (fal + CPU box) | Later (offline lab, 24 GB) | Fully-open fallback |
|---|---|---|---|
| Editor | FLUX.2 [dev] via fal | FLUX.2 dev GGUF (needs BFL commercial license for operational use) | Qwen-Image-Edit-2509 (Apache 2.0) — Phase D head-to-head; klein-4B (Apache) as small option |
| Detector | buffalo_l SCRFD (research-only) | same | SCRFD retrain/licensed — later decision |
| Rank embedder | buffalo_l ArcFace + AdaFace (MIT) | same | AdaFace only |
| SR | Real-ESRGAN x4plus (BSD-3), CPU ok | same, GPU | — |
| FIQA | SER-FIQ / CR-FIQA(S) (CPU) | same | — |
| Geometry prior (optional, Phase E) | — | 3DDFA_V3 (MIT) normals → conditioning experiment | — |
| Video SR (optional, Phase E) | — | KEEP (S-Lab NC — research only) / TIGER if weights appear | — |

## 5. Phases

- **A — Routing + Normalize + Portrait rendering + prompts** (pure code except
  renders). Acceptance: 16/16 New DataSet frames route correctly offline; old
  dataset routing unchanged; smoke renders on 2 subjects look like the target
  panels.
- **B — Evidence stage + harness + battery.** Real-ESRGAN path + cosine gate;
  `tools/test_new_dataset.py` (auto-split paired sheets, ArcFace vs target
  panel + vs fused profile, report.json + contact sheets); ablations R0 (no SR)
  vs R1 (SR) on 3 subjects; full battery with winner; old-dataset regression
  battery must not drop.
- **C — Multi-frame.** API accepts image list; FIQA triage; frame selection
  maximizes POSE DIVERSITY × quality (near-duplicate profile frames add ~no
  evidence — multi-view only pays when angles differ); filter-then-weight
  fusion; up to 4 refs to the editor. Acceptance: multi-frame ≥ single-frame on
  every subject with >1 usable frame.
- **D — Editor A/B + embedder swap + enrichment experiment.** Step 0: verify a
  Qwen-Image-Edit-2509 endpoint exists on fal (else run the A/B on the lab
  server). Then: Qwen vs FLUX.2 on the same harness (same refs/seeds/pool);
  AdaFace ranking on small faces; optional cross-model candidate pool if both
  are close. Experiment: reference enrichment — round 2 with the winning
  candidate ADDED as a third reference (distinct from the failed refine-edit,
  which fed it back as the image to edit); accepted only if score does not
  drop. Decides the long-term default + resolves the license question.
- **E — Lab deployment + research options.** Move winning stack to lab server;
  optional: 3DDFA_V3 geometry conditioning, KEEP for clips, DREAM-style
  pose-debiased ranking target.

Each phase ends with: battery numbers reported, commit on `Feature/Half_Face`,
push to github.

## 5b. Scope iteration findings (2026-08-03, 28 fal renders, 5 subjects)

Measured with tools/iterate_v3.py on the Scope set (2 far-CCTV with reference
targets, near-CCTV, phone profile, clean studio profile):

- crop + fixed 832×1216 + (descriptive, idportrait) pool renders proper
  ID-style portraits on all five inputs; idportrait and descriptive split the
  wins (keep both in the pool; "preserving" confirmed dropped on this route).
- vs-target (AI reference) picks land 0.38-0.42, oracles 0.42-0.51, while the
  raw CCTV profile itself anchors at only 0.30-0.49 — outputs score at or above
  the input's own similarity ceiling.
- SELECTION is the bottleneck, not generation. Two hard findings:
  (1) candidates scoring HIGHEST on min-both are often pose-biased clones —
  on 3 of 4 scored runs the oracle had a LOWER min_both than the pick; one
  candidate hit 0.888/frontality 0.0 (a still-in-profile render).
  (2) the frontality ≥ 0.55 + suspicious ≤ 0.74 gates catch exactly those —
  gate first, rank after, and surface top-3 to the user instead of trusting
  rank-1 (at CCTV anchor quality, ±0.05 of ranking signal is noise).
- Consensus ranking within a config picked the oracle 2/3; blended min_both +
  0.5·consensus underperformed — if consensus is used, use it as tiebreak
  among gated candidates, not blended.
- Auto-age injection from tiny faces is unreliable (37y read on a ~30y face);
  inject sex always, age only when the detected face is ≥200 px.
- Tight-face-only refs (tight-hs) did not beat crop+mirror refs; keep
  [crop, mirror] default, tight crop reserved for the ranking anchor.

Rounds 4-6 (same day, +22 renders — RECIPE LOCKED):

- idportrait2 (feature-list prompt) REJECTED by A/B: never won a subject —
  naming every axis pushed outputs generic. Pool stays (descriptive,
  idportrait); styles split wins per subject, keep both.
- ENRICHMENT VALIDATED, tier-conditional: one extra round, winning style,
  refs [crop, mirror, draft-winner] + "third image is a draft…correct any
  differing feature" sentence. CCTV tier: +0.10 (young8 0.41→0.51) or flat
  (man2 0.465→0.472); close-up tier: HURTS (old pair 0.606→0.592). Ship it
  only when face < 200 px. NEVER compare base vs enriched by min-both across
  rounds: enrichment lowers similarity-to-profile while improving identity —
  the enriched round's gated pick simply replaces the base pick in-tier.
- Previous-dataset regression (real ground truth, user's own pair): pool-8
  pick = oracle = 0.6329 vs v2 bar 0.636 (parity, within noise) and above
  the profile's own 0.6302 anchor; gates auto-caught a profile-clone render
  (min_both 0.87, frontality 0.0) that plain ranking would have surfaced.
- Final shipped numbers (picks, not oracles): man2 0.462-0.472 (input anchor
  0.303), young8 0.513 (anchor 0.485), prev-dataset 0.633 (anchor 0.630).
  Renders in Test_AI_Half/v3/final_board.png.

FINAL v3 RECIPE (wired into backend _frontalize_profile):
detect → head_shoulders_crop (tight crop = ranking anchor) → FLUX.2 edit at
832×1216, refs [crop, mirror], pool (descriptive, idportrait) × seeds →
gates (frontality ≥ 0.55, min_both ≤ 0.74) → pick by min_both among
survivors → if face < 200 px: enrichment round replaces pick (when its own
gated pick survives) → all candidates + tier + enriched flag to the UI;
sex auto-injected, age only when face ≥ 200 px.

## 5c. v4 — the 0.75 push: multi-photo fusion + the measured ceiling (2026-08-03)

Identity clustering over every available photo found MULTI-PHOTO SUBJECTS:
young cluster = young1/3/6/7/8 + clean_profile + near_cctv (7 photos, one
person); man cluster = man1/2/3; old cluster = old1/old3; nawaf = 1 photo.

Measured over 3 more rounds (~30 renders):
- Multi-photo references: a pile of scene crops DILUTES identity (man2 4-ref
  run fell to 0.28-0.30); density wins — primary head-shoulders crop + other
  photos as TIGHT face crops ("hs+tight"), ≤3-4 refs, multiref prompt styles.
- Fused quality-weighted anchor (all photos + mirrors) for ranking: picked
  the ORACLE on every multi-photo run. Wired into backend (`extra_images`).
- young8 with 3-photo evidence: pick = oracle = 0.5171 (single-photo best
  0.5127).
- THE CEILING MEASUREMENT that settles the 0.75 question: the AI reference
  targets themselves agree with the subjects' full photo evidence at only
  tgt_young8 = 0.527 (fused 5 photos), tgt_man2 = 0.366 (fused 3 photos).
  Our picks sit at 98% of young8's ceiling and ABOVE man2's. Scores beyond
  the ceiling measure agreement with the target-generator's hallucinations,
  not identity — 0.75 against these targets is mathematically unreachable
  for any honest reconstruction. On the real-truth benchmark (whatsapp),
  pools of 4/8/12 gave 0.6055/0.6329/0.6329 = saturation at the profile's
  own 0.630 anchor; single-profile information is exhausted.
- What a real 0.75 requires: REAL frontal ground-truth targets (retire the
  AI panels) and/or richer per-subject evidence (frames spanning genuinely
  different angles). The pipeline is ready for both (extra_images API).

## 6. Measurement protocol

Three buckets, one config must win across all (no per-domain tuning):
old WhatsApp close-ups (regression, has real ground truth) · New DataSet CCTV
sheets (target; right panel = reference bar, not ground truth) · phone-quality
profiles. Metrics: ArcFace/AdaFace cosine (output vs target, vs fused profile),
pixel-exact guarantee untouched on the mirror route, plus the user's eyes as
final judge. The harness also reports, per bucket, the rank correlation
between our selection score and similarity-to-target across the pool — the
ρ≈0.69 that justifies best-of-N was measured on close-ups and must be shown to
survive the CCTV regime, otherwise pool money is wasted and the ranker needs
fixing first. Budget note: pool of 8 ≈ 8 min/subject on fal; smoke subsets
before full batteries.
