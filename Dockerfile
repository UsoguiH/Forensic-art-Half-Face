FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/req.txt
RUN pip install --no-cache-dir -r /app/req.txt \
        "qdrant-client>=1.12" "minio>=7.2" "kaggle>=1.6" "diffusers>=0.39"

COPY backend/ /app/backend
COPY static/ /app/static
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENV FACELAB_STATIC_DIR=/app/static \
    FACELAB_INDEX_DIR=/data \
    FACELAB_DATASET_ROOT=/data/mugshots \
    PORT=8010
EXPOSE 8010

WORKDIR /app/backend
ENTRYPOINT ["/app/docker-entrypoint.sh"]
