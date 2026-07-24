#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"

if [[ "${SOC_VERSION}" != "Ascend950" &&
      ! "${SOC_VERSION}" =~ ^Ascend950(PR|DT)_[[:alnum:]_.-]+$ ]]; then
  echo "unsupported Ascend 950 SOC_VERSION: ${SOC_VERSION}" >&2
  echo "run scripts/check_soc_version.sh to obtain the full value" >&2
  exit 2
fi

if [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DPYTHON_BIN="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
