#!/usr/bin/env bash
set -e

export UV_YES=1

VENV_DIR=".venv"
KERNEL_NAME="eve-platform-tour"
KERNEL_DISPLAY_NAME="EVE Platform Tour"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${VENV_DIR}/bin/activate"

uv pip install -r "${SCRIPT_DIR}/requirements.txt" ipykernel

python -m ipykernel install --user --force \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
