#!/usr/bin/env bash
#
# Usage:
#   ./setup.sh                       # run every setup.sh under the repo
#   DAY=2 ./setup.sh                 # only Day 2
#   DAY=2 TRACK=1 ./setup.sh         # only Day 2 / Track 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERROR_FILE="${ROOT_DIR}/setup_errors.log"
LOG_FILE="${ROOT_DIR}/setup.log"

if [ -n "${DAY}" ] && [ -n "${TRACK}" ]; then
    SEARCH_DIR="${ROOT_DIR}/Day ${DAY}/Track ${TRACK}"
elif [ -n "${DAY}" ]; then
    SEARCH_DIR="${ROOT_DIR}/Day ${DAY}"
else
    SEARCH_DIR="${ROOT_DIR}"
fi

if [ ! -d "${SEARCH_DIR}" ]; then
    echo "error: directory not found: ${SEARCH_DIR}" >&2
    exit 1
fi

: > "${ERROR_FILE}"
: > "${LOG_FILE}"

run_setup() {
    local script="$1"
    local dir
    dir="$(dirname "${script}")"
    local name="${dir#${ROOT_DIR}/}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting: ${name}" >> "${LOG_FILE}"
    if ( cd "${dir}" && bash "${script}" ) >> "${LOG_FILE}" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ok: ${name}" >> "${LOG_FILE}"
    else
        local rc=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (exit ${rc}): ${name}" | tee -a "${ERROR_FILE}" >> "${LOG_FILE}"
    fi
}

main() {
    while IFS= read -r -d '' script; do
        [ "${script}" = "${ROOT_DIR}/setup.sh" ] && continue
        run_setup "${script}"
    done < <(find "${SEARCH_DIR}" -name setup.sh -type f -print0)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] all done" >> "${LOG_FILE}"
}

nohup bash -c "$(declare -f run_setup main); ROOT_DIR='${ROOT_DIR}' SEARCH_DIR='${SEARCH_DIR}' ERROR_FILE='${ERROR_FILE}' LOG_FILE='${LOG_FILE}' main" >/dev/null 2>&1 &

echo "setup running in background (pid $!)"
echo "scope:  ${SEARCH_DIR#${ROOT_DIR}/}"
echo "log:    ${LOG_FILE}"
echo "errors: ${ERROR_FILE}"
