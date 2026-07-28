# FaceLab — Text-to-Face Generation & Embedding-Based Retrieval

A generative-AI platform that synthesizes facial images from natural-language
descriptions, encodes them into 512-dimensional ArcFace embeddings, and retrieves
the most similar identities from a large-scale vector index.

## Features
- **Consensus search** — generate N faces from one description, manipulate their
  embeddings to cancel per-sample noise, and rank a large face index by cosine
  similarity, returning an interpretable similarity score.
- **Identity Studio** — generate a base face, lock its Vector embediing fingerprint, and
  apply identity-preserving edits and reconstruction of partial or angled faces.
- **Vector database (Qdrant)** — live ingestion of custom image datasets (zip or
  server folder) with immediate searchability.
- **Deployment** — containerized multi-service stack (Docker) or a cloud-GPU notebook.

## Repository layout
- `backend/`                 FastAPI service: generation, ArcFace embeddings, consensus search, galleries.
- `model_server/`            FLUX.2 generation API (served on port 8000).
- `static/index.html`        Single-page web interface.
- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`   Containerized stack.

## Quick start (Docker)
1. `cp .env.example .env` and fill in the MinIO password, Kaggle credentials, and HuggingFace token.
2. Place the prebuilt index (`front.index`, `front_ids.npy`, `front_paths.json`) in `./data`.
3. `docker compose up -d --build`
4. Open `http://<host>:8010`

## Requirements
NVIDIA GPU with the NVIDIA Container Toolkit, Docker, and a HuggingFace token with
access to the FLUX.2 models. The reference face dataset is downloaded automatically
on first boot via the Kaggle API when credentials are provided.

## Tech stack
Python · FastAPI · PyTorch · HuggingFace Diffusers (FLUX.2) · InsightFace / ArcFace ·
FAISS · Qdrant · MinIO · Docker
