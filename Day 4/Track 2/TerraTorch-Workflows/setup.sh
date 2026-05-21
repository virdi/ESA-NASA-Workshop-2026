#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="terratorch-workflows"
KERNEL_DISPLAY_NAME="TerraTorch Workflows"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_S3="s3://enw-04241552-kx1nks-shared/data/workshop_bundle.zip"
DATA_ROOT="${SCRIPT_DIR}"
BUNDLE_DIR="${DATA_ROOT}/workshop_bundle"

# HLS Burn Scars dataset (notebooks 01, 02, 04) and TerraMind embeddings (notebook 04).
BURNSCARS_DIR="${SCRIPT_DIR}/01_TerraTorch_Embeddings/hls_burn_scars"
BURNSCARS_ARCHIVE="${SCRIPT_DIR}/01_TerraTorch_Embeddings/hls_burn_scars.tar.gz"
BURNSCARS_GDRIVE_ID="1yFDNlGqGPxkc9lh9l1O70TuejXAQYYtC"
EMBEDDINGS_DIR="${BURNSCARS_DIR}/embeddings_terramind"
EMBEDDINGS_ZIP="${SCRIPT_DIR}/01_TerraTorch_Embeddings/hls_burn_scars_embeddings_terramind.zip"
EMBEDDINGS_GDRIVE_ID="1SA-WVWVC0d-s5BRKrup54lp7oFmjbSkR"
ITERATE_LINK="${SCRIPT_DIR}/04_Iterate_HPO_NAS/hls_burn_scars"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements.txt" ipykernel

# ~15 GB forest-disturbance bundle (notebook 03). Skip if already unzipped.
if [ ! -d "${BUNDLE_DIR}" ]; then
    mkdir -p "${DATA_ROOT}"
    aws s3 cp "${DATA_S3}" "${DATA_ROOT}/workshop_bundle.zip"
    unzip -q "${DATA_ROOT}/workshop_bundle.zip" -d "${DATA_ROOT}"
    rm -f "${DATA_ROOT}/workshop_bundle.zip"
fi

# HLS Burn Scars dataset (notebooks 01, 02, 04). Skip if already extracted.
if [ ! -d "${BURNSCARS_DIR}" ]; then
    mkdir -p "${SCRIPT_DIR}/01_TerraTorch_Embeddings"
    gdown "https://drive.google.com/uc?id=${BURNSCARS_GDRIVE_ID}" -O "${BURNSCARS_ARCHIVE}"
    tar -xzf "${BURNSCARS_ARCHIVE}" -C "${SCRIPT_DIR}/01_TerraTorch_Embeddings"
    rm -f "${BURNSCARS_ARCHIVE}"
fi

# Precomputed TerraMind embeddings (notebook 04). Skip if already extracted.
if [ ! -d "${EMBEDDINGS_DIR}" ]; then
    gdown "https://drive.google.com/uc?id=${EMBEDDINGS_GDRIVE_ID}" -O "${EMBEDDINGS_ZIP}"
    unzip -q "${EMBEDDINGS_ZIP}" -d "${BURNSCARS_DIR}"
    rm -f "${EMBEDDINGS_ZIP}"
fi

# Notebook 04 expects hls_burn_scars/ next to its own ipynb. Symlink to the canonical copy.
if [ ! -e "${ITERATE_LINK}" ]; then
    ln -s "../01_TerraTorch_Embeddings/hls_burn_scars" "${ITERATE_LINK}"
fi

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
