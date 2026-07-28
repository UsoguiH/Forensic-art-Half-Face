#!/usr/bin/env bash
# First-boot bootstrap: verify the search index, fetch the mugshot dataset from
# Kaggle if it is not present yet, then start the API.
set -uo pipefail

DATA_DIR="${FACELAB_INDEX_DIR:-/data}"
IMG_ROOT="${FACELAB_DATASET_ROOT:-/data/mugshots}"
KAGGLE_SLUG="${FACELAB_KAGGLE_SLUG:-elliotp/idoc-mugshots}"

echo "[bootstrap] index dir   = $DATA_DIR"
echo "[bootstrap] images root = $IMG_ROOT"

missing_index=0
for f in front.index front_ids.npy front_paths.json; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "[bootstrap] WARNING: missing index file  $DATA_DIR/$f"
    missing_index=1
  fi
done
if [ "$missing_index" = "1" ]; then
  echo "[bootstrap] Put front.index, front_ids.npy and front_paths.json in ./data."
  echo "[bootstrap] The API will still start; consensus search stays idle until the index is present."
else
  echo "[bootstrap] index OK — all three files present."
fi

have_images=0
if [ -d "$IMG_ROOT" ] && find "$IMG_ROOT" -type f -print -quit 2>/dev/null | grep -q .; then
  have_images=1
fi

if [ "$have_images" = "1" ]; then
  echo "[bootstrap] images already present at $IMG_ROOT — skipping download."
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  echo "[bootstrap] downloading '$KAGGLE_SLUG' from Kaggle into $IMG_ROOT (first boot only) ..."
  mkdir -p "$IMG_ROOT"
  if kaggle datasets download -d "$KAGGLE_SLUG" -p "$IMG_ROOT" --unzip; then
    n=$(find "$IMG_ROOT" -type f 2>/dev/null | wc -l)
    echo "[bootstrap] Kaggle download complete — $n files under $IMG_ROOT."
  else
    echo "[bootstrap] WARNING: Kaggle download failed. Check KAGGLE_USERNAME / KAGGLE_KEY and network."
  fi
else
  echo "[bootstrap] no images at $IMG_ROOT and no KAGGLE_USERNAME / KAGGLE_KEY set."
  echo "[bootstrap] Either mount the images at $IMG_ROOT or set Kaggle credentials in .env."
fi

export FACELAB_DATASET_ROOT="$IMG_ROOT"

echo "[bootstrap] starting API on :${PORT:-8010}"
exec python -m uvicorn backend:app --host 0.0.0.0 --port "${PORT:-8010}"
