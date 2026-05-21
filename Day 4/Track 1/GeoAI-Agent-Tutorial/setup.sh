#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive
export UV_YES=1

REPO_URL="https://github.com/NASA-IMPACT/fm-inference-sagemaker.git"
REPO_DIR="fm-inference-sagemaker"
FM_VENV_DIR=".venv-fm-inference"
TUTORIAL_VENV_DIR=".venv"
KERNEL_NAME="geoai-agent"
KERNEL_DISPLAY_NAME="GeoAI Agent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/gunicorn.pid"

export BUCKET_NAME="${BUCKET_NAME:-enw-04241552-kx1nks-shared}"
export S3_CONFIG_FILENAME="${S3_CONFIG_FILENAME:-s3://enw-04241552-kx1nks-shared/geo-agent-artifacts/config.yaml}"
export CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-s3://enw-04241552-kx1nks-shared/geo-agent-artifacts/Prithvi-EO-V2-600-Sen1Floods11.pt}"
export USECASE="${USECASE:-flood}"
export MODEL_SERVER_TIMEOUT="${MODEL_SERVER_TIMEOUT:-150}"

if [ -z "${S3_CONFIG_FILENAME}" ] || [ -z "${CHECKPOINT_FILENAME}" ]; then
    echo "S3_CONFIG_FILENAME and CHECKPOINT_FILENAME must be set" >&2
    exit 1
fi

sudo -n apt-get update -y
sudo -n apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" libgl1 libglib2.0-0 libgdal-dev

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d "${SCRIPT_DIR}/${REPO_DIR}" ]; then
    git clone "${REPO_URL}" "${SCRIPT_DIR}/${REPO_DIR}"
fi

uv venv --python 3.12 --allow-existing "${SCRIPT_DIR}/${FM_VENV_DIR}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${FM_VENV_DIR}/bin/activate"
uv pip install -r "${SCRIPT_DIR}/${REPO_DIR}/requirements.txt"

mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}/${REPO_DIR}/code"

timeout="${MODEL_SERVER_TIMEOUT}"
host="${MODEL_SERVER_HOST:-0.0.0.0}"
port="${MODEL_SERVER_PORT:-8080}"

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    sleep 1
fi

nohup gunicorn \
    --timeout "${timeout}" \
    -b "${host}:${port}" \
    -w 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile "${LOG_DIR}/access.log" \
    --error-logfile "${LOG_DIR}/error.log" \
    --capture-output \
    predictor:app \
    >"${LOG_DIR}/gunicorn.out" 2>&1 &
echo $! >"${PID_FILE}"
disown

deactivate

cd "${SCRIPT_DIR}"

uv sync --frozen

# akd → crawl4ai pulls in `unclecode-litellm`, a soft-fork of litellm that
# installs into the same `litellm/` directory as the official package. On some
# installs (notably SageMaker) the two overwrite each other file-by-file,
# producing a Frankenstein litellm where `litellm/types/utils.py` (from 1.83.0)
# imports symbols that `litellm/types/llms/openai.py` (from the older fork)
# doesn't define — e.g. `ChatCompletionReasoningItem`. Nothing in this tutorial
# actually uses crawl4ai, so we drop the fork and re-extract the official
# litellm to leave the install internally consistent.
uv pip uninstall unclecode-litellm
uv pip install --force-reinstall --no-deps litellm==1.83.0

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/${TUTORIAL_VENV_DIR}/bin/activate"

python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
