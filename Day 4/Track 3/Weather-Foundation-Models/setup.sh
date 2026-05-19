#!/bin/bash
# ESA-NASA Workshop Environment Setup Script — Weather Foundation Models
# Creates a uv-managed venv, registers a Jupyter kernel, and pre-downloads
# the Prithvi-WxC inputs/weights referenced by the rollout notebook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRITHVI_DIR="${TRACK_DIR}/Prithvi-WxC"
DATA_DIR="${PRITHVI_DIR}/data"
WEIGHTS_DIR="${DATA_DIR}/weights"

VENV_DIR="${TRACK_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

PYTHON_VERSION="3.11"
KERNEL_NAME="weather-fm"
KERNEL_DISPLAY_NAME="Python (Weather FM)"

PRITHVI_WXC_REPO="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"
PRITHVI_WXC_ROLLOUT_REPO="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout"

CHECKPOINTS_S3="s3://enw-04241552-kx1nks-shared/data/checkpoints/"
CHECKPOINTS_DIR="${TRACK_DIR}/data/checkpoints"

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

hf download "${PRITHVI_WXC_REPO}" \
    --include "merra-2/MERRA2_sfc_2020010[1-6].nc" \
    --local-dir "${DATA_DIR}"
hf download "${PRITHVI_WXC_REPO}" \
    --include "merra-2/MERRA_pres_2020010[1-6].nc" \
    --local-dir "${DATA_DIR}"

hf download "${PRITHVI_WXC_REPO}" \
    --include "climatology/climate_surface_doy00[1-6]*.nc" \
    --local-dir "${DATA_DIR}"
hf download "${PRITHVI_WXC_REPO}" \
    --include "climatology/climate_vertical_doy00[1-6]*.nc" \
    --local-dir "${DATA_DIR}"

hf download "${PRITHVI_WXC_REPO}" \
    climatology/musigma_surface.nc \
    climatology/musigma_vertical.nc \
    climatology/anomaly_variance_surface.nc \
    climatology/anomaly_variance_vertical.nc \
    --local-dir "${DATA_DIR}"

hf download "${PRITHVI_WXC_ROLLOUT_REPO}" config.yaml --local-dir "${DATA_DIR}"
hf download "${PRITHVI_WXC_ROLLOUT_REPO}" \
    prithvi.wxc.rollout.2300m.v1.pt \
    --local-dir "${WEIGHTS_DIR}"

# 6. Pull workshop checkpoints from S3 (shared bucket).
mkdir -p "${CHECKPOINTS_DIR}"
aws s3 sync --size-only "${CHECKPOINTS_S3}" "${CHECKPOINTS_DIR}/"

# Note: GraphCast pulls its weights and sample data from the public
# gs://dm_graphcast bucket from inside graphcast_demo.ipynb. No pre-download
# step here.

echo
echo "Setup complete."
echo "  venv:    ${VENV_DIR}"
echo "  kernel:  ${KERNEL_DISPLAY_NAME} (${KERNEL_NAME})"
echo "  data:        ${DATA_DIR}"
echo "  weights:     ${WEIGHTS_DIR}"
echo "  checkpoints: ${CHECKPOINTS_DIR}"
echo "Pick '${KERNEL_DISPLAY_NAME}' from the JupyterLab kernel picker."
