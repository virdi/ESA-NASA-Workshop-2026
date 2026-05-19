#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="terratorch-workflows"
KERNEL_DISPLAY_NAME="TerraTorch Workflows"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_S3="s3://enw-04241552-kx1nks-shared/data/workshop_bundle.zip"
DATA_ROOT="${WORKSHOP_DATA_ROOT:-${HOME}/workshop_data}"
BUNDLE_DIR="${DATA_ROOT}/workshop_bundle"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements.txt" ipykernel

# ~15 GB forest-disturbance bundle. Skip if already unzipped.
if [ ! -d "${BUNDLE_DIR}" ]; then
    mkdir -p "${DATA_ROOT}"
    aws s3 cp "${DATA_S3}" "${DATA_ROOT}/workshop_bundle.zip"
    unzip -q "${DATA_ROOT}/workshop_bundle.zip" -d "${DATA_ROOT}"
    rm -f "${DATA_ROOT}/workshop_bundle.zip"
fi

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
