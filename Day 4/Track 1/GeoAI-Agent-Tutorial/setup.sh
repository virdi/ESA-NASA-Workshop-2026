#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/NASA-IMPACT/fm-inference-sagemaker.git"
REPO_DIR="fm-inference-sagemaker"
FM_VENV_DIR=".venv-fm-inference"
TUTORIAL_VENV_DIR=".venv"
KERNEL_NAME="geoai-agent"
KERNEL_DISPLAY_NAME="GeoAI Agent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/gunicorn.pid"

sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libgdal-dev

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d "${SCRIPT_DIR}/${REPO_DIR}" ]; then
    git clone "${REPO_URL}" "${SCRIPT_DIR}/${REPO_DIR}"
fi

uv venv --python 3.11 "${SCRIPT_DIR}/${FM_VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${FM_VENV_DIR}/bin/activate"
uv pip install -r "${SCRIPT_DIR}/fm-inference-requirements.txt"

mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}/${REPO_DIR}/code"

timeout="${MODEL_SERVER_TIMEOUT:-60}"

gunicorn \
    --daemon \
    --pid "${PID_FILE}" \
    --timeout "${timeout}" \
    -b unix:/tmp/gunicorn.sock \
    -w 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile "${LOG_DIR}/access.log" \
    --error-logfile "${LOG_DIR}/error.log" \
    --capture-output \
    predictor:app

deactivate

cd "${SCRIPT_DIR}"

uv sync --frozen

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${TUTORIAL_VENV_DIR}/bin/activate"

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
