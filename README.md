# FaceLab - Text-to-Face Generation & Embedding-Based Retrieval

A generative-AI platform that synthesizes facial images from natural-language
descriptions, encodes them into 512-dimensional ArcFace embeddings, and retrieves
the most similar identities from a large-scale vector index.

## Features
- **Consensus search** - generate N faces from one description, manipulate their
  embeddings to cancel per-sample noise, and rank a large face index by cosine
  similarity, returning an interpretable similarity score.
- **Identity Studio** - generate a base face, lock its ArcFace embedding fingerprint,
  and apply identity-preserving edits and reconstruction of partial or angled faces.
- **Half-face completion** - hand it one half of a frontal portrait and get the whole
  face back, with the supplied half returned byte-for-byte identical. See below.
- **Vector database (Qdrant)** - live ingestion of custom image datasets (zip or
  server folder) with immediate searchability.
- **Deployment** - containerized stack (Docker) or a cloud-GPU notebook.

## Repository layout
- `backend/`                 FastAPI service: face generation, ArcFace embeddings, consensus search, Identity Studio, galleries.
- `backend/half_face.py`     Pixel-exact half-face completion (geometry, registration, seam, verification) + input classification (half / profile / full).
- `backend/frontalize.py`    Profile→frontal identity reconstruction (dual-reference candidates, ArcFace best-of-N).
- `backend/fal_provider.py`  fal.ai FLUX.2 client.
- `tools/test_half_face.py`  Scores half-face completion against the photos in `half photo/`.
- `static/index.html`        Single-page web interface.
- `data/`                    Prebuilt search index (front.index, front_ids.npy, front_paths.json).
- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`   Containerized stack.

## Generation routes

Where the pixels come from is one setting, `FACELAB_PROVIDER`. Nothing else in
the app changes when you switch.

| `FACELAB_PROVIDER` | Draws images with | Needs | Use it when |
|---|---|---|---|
| `fal` | FLUX.2 \[dev] on fal.ai | internet + `FAL_KEY` | you want the sharpest editing, and half-face completion |
| `openrouter` | hosted models via the OpenRouter API | internet + `OPENROUTER_API_KEY` | developing or demoing on a machine with no GPU |
| `lab` | the FLUX.2 server at `FLUX_API_URL` | that server reachable on port 8000 | on the offline lab network |
| `local` | a diffusers pipeline in this container | NVIDIA GPU + FLUX.2 weights | one box with a big GPU |

The hosted routes are not interchangeable with `lab` per environment: the lab
network has no internet, so `fal` and `openrouter` cannot work there, and `lab`
only works where that server is reachable. Left unset, the route is
auto-detected — a fal key wins, then an OpenRouter key, then `FLUX_API_URL`,
then local CUDA.

`GET /health` reports the active route under `provider`, plus per-route detail:
`fal` (endpoints and images rendered) or `openrouter` (models and credits spent).

**fal.ai.** Set `FAL_KEY=<id>:<secret>`, or drop the key in `data/fal_key.txt`
(that directory is gitignored). Editing goes to `fal-ai/flux-2/edit` — FLUX.2
\[dev] — and falls back to `fal-ai/flux-2/klein/9b/edit` if the account cannot
reach it; override either with `FAL_EDIT_MODEL`, or per request with
`"model": "dev" | "klein"`. This is the only route that both accepts reference
images and honours an exact pixel `image_size`, which is why half-face
completion prefers it even when another provider is the global default.

**OpenRouter caveats.** Its image endpoint takes no `seed`, so requests are not
reproducible — `steps`, `guidance`, and `seed` are accepted and ignored on that
route. Consensus search still works, because the spread it averages over comes
from sampling variation rather than distinct seeds. Bursts are batched 10 per
request, and pixel sizes are mapped onto OpenRouter's `aspect_ratio` +
`resolution` tiers.

## Half-face completion

`POST /half_face` takes one half of a frontal portrait and returns the whole
face. The half you supply comes back **byte-for-byte identical** — the response
says so in `pixel_exact`, and the server raises rather than return a result that
fails the check.

That guarantee is the whole point, and it is why this is not just a prompt. A
diffusion edit repaints the entire canvas, so the half you actually have
evidence for comes back resampled and re-lit. Here the model only ever supplies
the *missing* side:

1. the supplied half plus its mirror image become a full canvas, which hands the
   model correct head size, skin tone, hair and background to work from;
2. FLUX.2 renders it;
3. the render is re-registered against the supplied half — scale and
   translation are measured, not assumed, because a repainting model also
   re-frames by a few percent and a drifted seam is a visible seam;
4. the generated half is tone-matched, and its seam-adjacent strip is faded into
   the mirror, which is both the best estimate available at the midline and
   exactly continuous with the pixels about to be pasted back;
5. the supplied half is pasted over the result at its original coordinates.

Request: `image` (data URL) plus optional `side`
(`auto` *(default)* `| left | right | top | bottom` — which half you supplied),
`gender`, `model`, `seed`, `steps`, `guidance`, `seam_band`, `tone_match`,
`align_radius`, `align_scale`, `identity_anchor`. Auto side detection mirrors
the half both ways and asks the face detector which one is a face; it also
handles a full-size canvas with one half masked out rather than cropped away.

Pass `take: "left"` with a **complete** photo and the server keeps only that
half before completing it, then scores the reconstruction against the original
it discarded — that is the demo path, and the UI's "صورة كاملة — اقطعها واختبر"
checkbox.

### What a profile upload does instead

Uploads are classified before any mirroring (`method` in the response says
which route ran). A **side-profile photograph is not half of a frontal
portrait** — mirroring one builds a two-faced canvas, which the edit model
faithfully keeps. Profiles therefore take the `backend/frontalize.py` route:
several frontal candidates are rendered (the profile *and its mirror* ride
along as references so the model sees both sides of the head), every candidate
is ranked by ArcFace similarity to the profile, and the best one is returned.
`pixel_exact` is honestly `false` on this route — the front of a face is not
present in a profile photo, so it is reconstructed, not copied.

Measured on the ground-truth pairs in `half photo/` (a profile and a real
frontal of the same person): the selected candidate scored **0.636** ArcFace
against the person's real frontal — slightly *above* the 0.630 that the real
profile photo itself scores against that frontal, i.e. the reconstruction
extracts essentially all the identity information the profile carries.
`candidates` (default 6, max 8) trades cost for quality; `refine` (default
off) is an experimental second edit pass that so far *hurts* identity and is
only accepted when its score does not drop.

An upload that is already a **complete frontal face** comes back unchanged
with `method: "already-complete"`.

Response: `image`, `signature`, `pixel_exact`, `geometry`, `align`,
`seam_step` (0 = invisible join), `identity`, and — when a ground truth was
available — `accuracy`.

### Validating it

`tools/test_half_face.py` cuts every frontal photo in `half photo/` down the
middle, rebuilds each side, and scores the result against the half it hid:

```bash
python tools/test_half_face.py                # both halves of every frontal photo
python tools/test_half_face.py --sides left   # half the API calls
python tools/test_half_face.py --model klein  # FLUX.2 [klein] 9B instead of [dev]
```

Images and `report.json` land in `data/half_face_tests/`. Profile shots are
skipped automatically — completion assumes a frontal photo cut near the facial
midline.

Over the eight runs on that folder (four frontal photos × both halves):

| | result |
|---|---|
| supplied half returned unchanged | **8/8** |
| seam step across the join | **0.0** on every run |
| generated half vs the real one | 17.5 dB PSNR (13.4 – 21.5) |
| ArcFace, rebuilt vs original | 0.62 |
| ArcFace, plain mirror vs original | 0.62 |

Read those last two together. A mirrored face is a strong identity baseline
precisely because it is the real half twice — it is just obviously fake to look
at. Completion matches it on identity while producing a naturally asymmetric
face, which is the point; it does not beat it, and claiming otherwise would be
reading noise, since repeat runs at a fixed seed vary by roughly ±0.01. PSNR
against the hidden half is capped by information that genuinely is not in the
input (background detail, the exact hair strands), so treat it as a
regression check rather than a target.

Guidance defaults to 1.5 rather than the usual 2.5 (`half_face.DEFAULT_GUIDANCE`):
measured over the same folder, low guidance held identity better (0.59 vs 0.54)
and halved how far the model re-framed the canvas. `--model klein` runs about
40% faster and stays pixel-exact, but scored ~0.06 lower on identity.

## Quick start (Docker)
1. `cp .env.example .env`, then set `FAL_KEY` (or `OPENROUTER_API_KEY`) and the MinIO password.
2. Place the prebuilt index (`front.index`, `front_ids.npy`, `front_paths.json`) in `./data`.
3. `docker compose up -d --build`
4. Open `http://<host>:8010`

## Requirements
Docker. A GPU is required only for the `local` route — on `openrouter` and `lab`
all diffusion happens elsewhere, and the container needs a GPU only if you want
CUDA-accelerated ArcFace embeddings (they fall back to CPU). The HuggingFace
token is likewise only needed for `local`. The reference face dataset is
downloaded automatically on first boot via the Kaggle API when credentials are
provided.

## Tech stack
Python - FastAPI - PyTorch - HuggingFace Diffusers (FLUX.2) - InsightFace / ArcFace -
FAISS - Qdrant - MinIO - Docker
