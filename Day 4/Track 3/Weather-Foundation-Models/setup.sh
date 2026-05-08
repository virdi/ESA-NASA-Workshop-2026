#!/bin/bash
# ESA-NASA Workshop Environment Setup Script
# Creates a uv-managed venv for the Weather Foundation Models track and
# registers it as a Jupyter kernel for SageMaker JupyterLab.

set -euo pipefail

REPOSITORY_NAME="ESA-NASA-workshop-2026"
REPOSITORY_PATH="./${REPOSITORY_NAME}"

TRACK_DIR="${REPOSITORY_PATH}/Track 3/Weather-Foundation-Models"
PRITHVI_WX_WEIGHTS_DIR="${TRACK_DIR}/data/weights/"
VENV_DIR="${TRACK_DIR}/.venv"
REQUIREMENTS_FILE="./requirements.txt"

PYTHON_VERSION="3.11"
KERNEL_NAME="weather-fm"
KERNEL_DISPLAY_NAME="Python (Weather FM)"

# 1. Install uv if it's not already on PATH
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

# 2. Create the venv (uv will fetch the requested Python if missing).
#    --seed installs pip/setuptools/wheel so plain `pip install ...` also works.
uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
export VIRTUAL_ENV="${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# 3. Install the full stack from requirements.txt.
#    flash_attn and timm need --no-build-isolation; uv's requirements.txt
#    parser doesn't accept that per-line, so we set it via env var here.
export UV_NO_BUILD_ISOLATION_PACKAGE="flash-attn timm nvidia-pyindex"
uv pip install -r "${REQUIREMENTS_FILE}"

# 4. Register a Jupyter kernel that points at this venv
python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "Setup complete."
echo "  venv:   ${VENV_DIR}"
echo "  kernel: ${KERNEL_DISPLAY_NAME} (${KERNEL_NAME})"
echo "Pick '${KERNEL_DISPLAY_NAME}' from the JupyterLab kernel picker."
