#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="vllm-geoai"
KERNEL_DISPLAY_NAME="vLLM GeoAI"
MODEL_DIR="1_run_model_in_vllm"
HF_MODEL_REPO="mgazz/prithvi-eo-flood"
HF_BURNSCARS_REPO="mgazz/prithvi-eo-burnscars"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements.txt"

hf download "${HF_MODEL_REPO}" config_deploy.yaml --local-dir "${SCRIPT_DIR}/${MODEL_DIR}"
hf download "${HF_MODEL_REPO}" state_dict.ckpt --local-dir "${SCRIPT_DIR}/${MODEL_DIR}"
hf download "${HF_BURNSCARS_REPO}" park_fire_scaled.tif --local-dir "${SCRIPT_DIR}/samples"

python -m ipykernel install --user --force \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
