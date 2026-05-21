#!/bin/bash
# ESA-NASA Workshop Environment Setup Script — Weather Foundation Models
# Creates a uv-managed venv, registers a Jupyter kernel, and pre-downloads
# the Prithvi-WxC inputs/weights referenced by the rollout notebook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="${SCRIPT_DIR}"

# Notebooks hardcode these EFS paths; setup.sh writes to the same locations
# so the rollout/demo notebooks find the pre-staged files.
EFS_ROOT="${HOME}/user-default-efs"
EFS_CHECKPOINTS="${EFS_ROOT}/checkpoints"
PRITHVI_ROOT="${EFS_CHECKPOINTS}/prithvi"
DATA_DIR="${PRITHVI_ROOT}/data"
WEIGHTS_DIR="${DATA_DIR}/weights"

if [ ! -d "${EFS_ROOT}" ]; then
    echo "ERROR: EFS not mounted at ${EFS_ROOT}" >&2
    exit 1
fi

VENV_DIR="${TRACK_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.txt"

PYTHON_VERSION="3.11"
KERNEL_NAME="weather-fm"
KERNEL_DISPLAY_NAME="Python (Weather FM)"

PRITHVI_WXC_REPO="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"
PRITHVI_WXC_ROLLOUT_REPO="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout"

BUCKET_NAME="s3://enw-04241552-kx1nks-shared"

CHECKPOINTS_S3="${BUCKET_NAME}/data/checkpoints/"

export UV_YES=1

# 1. Install uv if needed.
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

# 2. Create the venv (uv fetches Python if missing) and activate it.
uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
export VIRTUAL_ENV="${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# 3. Install requirements.
#    flash_attn / timm / nvidia-pyindex need --no-build-isolation; uv reads
#    that from UV_NO_BUILD_ISOLATION_PACKAGE since requirements.txt can't
#    carry per-line flags.
export UV_NO_BUILD_ISOLATION_PACKAGE="flash-attn timm nvidia-pyindex"
uv pip install -r "${REQUIREMENTS_FILE}" ipykernel

# 4. Register the Jupyter kernel for SageMaker JupyterLab.
python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

# 5. Pre-download Prithvi-WxC data, climatology, and rollout weights.
#    Mirrors Prithvi-WxC/validation/get_assets.py — MERRA-2 sample window
#    2020-01-01..2020-01-06 plus the climatology and scaler files the
#    rollout config.yaml references.
mkdir -p "${DATA_DIR}/merra-2" "${DATA_DIR}/climatology" "${WEIGHTS_DIR}"

# 6. Pull workshop checkpoints from S3 (GraphCast + AIFS) into the EFS path
#    the GraphCast notebook reads from. The FMAifs notebook reads
#    ../checkpoints/aifs_single_v0.2.1.ckpt relative to its own directory,
#    so symlink ${TRACK_DIR}/checkpoints to the same EFS location.
mkdir -p "${EFS_CHECKPOINTS}"
aws s3 sync --size-only "${CHECKPOINTS_S3}" "${EFS_CHECKPOINTS}/"
ln -sfn "${EFS_CHECKPOINTS}" "${TRACK_DIR}/checkpoints"

aws s3 sync --size-only "${BUCKET_NAME}/data/stats" "${EFS_CHECKPOINTS}/"
ln -sfn "${EFS_CHECKPOINTS}" "${TRACK_DIR}/stats"

echo
echo "Setup complete."
echo "  venv:        ${VENV_DIR}"
echo "  kernel:      ${KERNEL_DISPLAY_NAME} (${KERNEL_NAME})"
echo "  prithvi:     ${PRITHVI_ROOT}"
echo "  checkpoints: ${EFS_CHECKPOINTS}"
echo "Pick '${KERNEL_DISPLAY_NAME}' from the JupyterLab kernel picker."
