FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/req.txt
RUN pip install --no-cache-dir -r /app/req.txt \
        "qdrant-client>=1.12" "minio>=7.2" "kaggle>=1.6" "diffusers>=0.39"

# Bake the ArcFace/RetinaFace models (buffalo_l) into the image so the first
# boot needs no internet — required for the offline lab deployment.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

COPY backend/ /app/backend
COPY static/ /app/static
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
# Strip Windows CR line endings (a Windows git checkout otherwise breaks the
# entrypoint with "/usr/bin/env: 'bash\r': No such file or directory").
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

ENV FACELAB_STATIC_DIR=/app/static \
    FACELAB_INDEX_DIR=/data \
    FACELAB_DATASET_ROOT=/data/mugshots \
    PORT=8010
EXPOSE 8010

WORKDIR /app/backend
ENTRYPOINT ["/app/docker-entrypoint.sh"]
