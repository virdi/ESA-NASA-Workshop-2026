#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="earth-embeddings-eo"
KERNEL_DISPLAY_NAME="Earth Embeddings EO"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRITHVI="ibm-nasa-geospatial/Prithvi-EO-2.0-300M"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements-gpu.txt" ipykernel

hf download "${PRITHVI}" prithvi_mae.py config.json Prithvi_EO_V2_300M.pt
hf download "${PRITHVI}" \
    examples/Mexico_HLS.S30.T13REM.2018026T173609.v2.0_cropped.tif \
    examples/Mexico_HLS.S30.T13REM.2018106T172859.v2.0_cropped.tif \
    examples/Mexico_HLS.S30.T13REM.2018201T172901.v2.0_cropped.tif \
    examples/Mexico_HLS.S30.T13REM.2018266T173029.v2.0_cropped.tif

hf download zterrabyte/eurosat-gitrsclip-embeddings --repo-type dataset eurosat_test_gitrsclip.npz
hf download zterrabyte/levircd-changeclip-embeddings --repo-type dataset levircd_changeclip_pairs.npz
hf download FuxunTB/changetx-text-change-demo --repo-type dataset

python -m ipykernel install --user --force \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
