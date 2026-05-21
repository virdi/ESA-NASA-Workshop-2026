#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="terratorch-workflows"
KERNEL_DISPLAY_NAME="TerraTorch Workflows"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bulk data lives on EFS; the script dir holds only symlinks so the
# 100 GB root volume only carries the venv.
EFS_ROOT="${HOME}/user-default-efs"
EFS_DATA="${EFS_ROOT}/terratorch"

DATA_S3="s3://enw-04241552-kx1nks-shared/data/workshop_bundle.zip"
EFS_BUNDLE="${EFS_DATA}/workshop_bundle"
BUNDLE_LINK="${SCRIPT_DIR}/workshop_bundle"

BURNSCARS_GDRIVE_ID="1yFDNlGqGPxkc9lh9l1O70TuejXAQYYtC"
EMBEDDINGS_GDRIVE_ID="1SA-WVWVC0d-s5BRKrup54lp7oFmjbSkR"
EFS_BURNSCARS="${EFS_DATA}/hls_burn_scars"
BURNSCARS_LINK="${SCRIPT_DIR}/01_TerraTorch_Embeddings/hls_burn_scars"
ITERATE_LINK="${SCRIPT_DIR}/04_Iterate_HPO_NAS/hls_burn_scars"

if [ ! -d "${EFS_ROOT}" ]; then
    echo "ERROR: EFS not mounted at ${EFS_ROOT}" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements.txt" ipykernel

mkdir -p "${EFS_DATA}"

# ~15 GB forest-disturbance bundle (notebook 03). Skip if already unzipped.
if [ ! -d "${EFS_BUNDLE}" ]; then
    aws s3 cp "${DATA_S3}" "${EFS_DATA}/workshop_bundle.zip"
    unzip -q "${EFS_DATA}/workshop_bundle.zip" -d "${EFS_DATA}"
    rm -f "${EFS_DATA}/workshop_bundle.zip"
fi
ln -sfn "${EFS_BUNDLE}" "${BUNDLE_LINK}"

# HLS Burn Scars dataset (notebooks 01, 02, 04). Skip if already extracted.
if [ ! -d "${EFS_BURNSCARS}" ]; then
    archive="${EFS_DATA}/hls_burn_scars.tar.gz"
    gdown "https://drive.google.com/uc?id=${BURNSCARS_GDRIVE_ID}" -O "${archive}"
    tar -xzf "${archive}" -C "${EFS_DATA}"
    rm -f "${archive}"
fi
mkdir -p "${SCRIPT_DIR}/01_TerraTorch_Embeddings" "${SCRIPT_DIR}/04_Iterate_HPO_NAS"
ln -sfn "${EFS_BURNSCARS}" "${BURNSCARS_LINK}"
ln -sfn "${EFS_BURNSCARS}" "${ITERATE_LINK}"

# Precomputed TerraMind embeddings (notebook 04). Skip if already extracted.
if [ ! -d "${EFS_BURNSCARS}/embeddings_terramind" ]; then
    zip="${EFS_DATA}/hls_burn_scars_embeddings_terramind.zip"
    gdown "https://drive.google.com/uc?id=${EMBEDDINGS_GDRIVE_ID}" -O "${zip}"
    unzip -q "${zip}" -d "${EFS_BURNSCARS}"
    rm -f "${zip}"
fi

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
