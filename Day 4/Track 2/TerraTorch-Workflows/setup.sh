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
DATA_LINK="${SCRIPT_DIR}/data"
AUDIT_LINK="${SCRIPT_DIR}/audit"

BURNSCARS_S3="s3://enw-04241552-kx1nks-shared/day4/hls_burn_scars.tar.gz"
EMBEDDINGS_S3="s3://enw-04241552-kx1nks-shared/day4/hls_burn_scars_embeddings_terramind.zip"
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

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

mkdir -p "${EFS_DATA}"

# HLS Burn Scars dataset (notebooks 01, 02, 04). Skip if already extracted.
if [ ! -d "${EFS_BURNSCARS}" ]; then
    archive="${EFS_DATA}/hls_burn_scars.tar.gz"
    aws s3 cp "${BURNSCARS_S3}" "${archive}"
    tar -xzkf "${archive}" -C "${EFS_DATA}"
    rm -f "${archive}"
fi
mkdir -p "${SCRIPT_DIR}/01_TerraTorch_Embeddings" "${SCRIPT_DIR}/04_Iterate_HPO_NAS"
ln -sfn "${EFS_BURNSCARS}" "${BURNSCARS_LINK}"
ln -sfn "${EFS_BURNSCARS}" "${ITERATE_LINK}"

# Precomputed TerraMind embeddings (notebook 04). The zip has a top-level
# hls_burn_scars/ wrapper, so extract into EFS_DATA so it merges with the
# existing hls_burn_scars/ instead of nesting.
if [ ! -d "${EFS_BURNSCARS}/embeddings_terramind" ]; then
    zip="${EFS_DATA}/hls_burn_scars_embeddings_terramind.zip"
    aws s3 cp "${EMBEDDINGS_S3}" "${zip}"
    unzip -nq "${zip}" -d "${EFS_DATA}"
    rm -f "${zip}"
fi

# ~15 GB forest-disturbance bundle (notebook 03). Skip if already unzipped.
if [ ! -d "${EFS_BUNDLE}" ]; then
    bundle="${EFS_DATA}/workshop_bundle.zip"
    aws s3 cp "${DATA_S3}" "${bundle}"
    echo "Extracting workshop_bundle.zip onto EFS (~30 GB, expect 5-15 min)..."
    time unzip -nq "${bundle}" -d "${EFS_DATA}"
    echo "Extraction complete."
    rm -f "${bundle}"
fi
# Notebook 03 reads ../data and ../audit (auto-derived as siblings); point
# those at the bundle's subdirs so no SageMaker-vs-local branching is needed.
ln -sfn "${EFS_BUNDLE}/data" "${DATA_LINK}"
ln -sfn "${EFS_BUNDLE}/audit" "${AUDIT_LINK}"
