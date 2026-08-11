# Deploying FaceLab on the offline lab machine (Docker Compose)

The lab network has no internet, so the images are **built where internet
exists, then moved as files**. Runtime needs no internet: generation goes to
the labmate's FLUX server on port 8000, and the face-embedding models are
baked into the image at build time.

## 1. Build (any machine WITH internet and Docker)

```bash
cd Forensic_Art
docker compose build            # builds facelab:latest (large: CUDA + torch)
docker compose pull qdrant minio
```

## 2. Export the three images to one file

```bash
docker save -o facelab-stack.tar \
  facelab:latest qdrant/qdrant:v1.12.4 minio/minio:latest
```

The file is about 4.6 GB (layers are stored compressed). Move
`facelab-stack.tar` to the lab machine with a USB drive (or upload through
Jupyter if the drive is not an option).

## 3. Prepare the folder on the lab machine

Copy the project folder (this repo) to the lab machine, then inside it:

- `cp .env.example .env` and set the MinIO password. Keep
  `FLUX_API_URL=http://host.docker.internal:8000` and leave the Kaggle keys
  empty (no internet).
- Put the prebuilt index in `./data/`: `front.index`, `front_ids.npy`,
  `front_paths.json`.
- Put the mugshot images in `./data/mugshots/` (copied, not downloaded).

## 4. Load and start (on the lab machine)

```bash
docker load -i facelab-stack.tar
docker compose up -d            # NO --build: it must use the loaded image
docker compose logs -f facelab  # watch the bootstrap lines
```

The labmate's FLUX server must be running and published on port 8000
(`curl http://localhost:8000/health` answers).

## 5. Verify

```bash
curl http://localhost:8010/health
```

The reply must contain `"engine": "remote"` — that confirms FaceLab will send
all generation to the FLUX server. Then from any PC on the lab network open:

    http://192.168.100.127:8010

Generate a face in the UI; the image is produced by the FLUX server's GPU.

## Requirements on the lab machine

- Docker with the NVIDIA Container Toolkit (already true — the FLUX server
  runs on GPU in Docker there).
- Port 8010 free (FaceLab UI/API), 9001 free (MinIO console).
