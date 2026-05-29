#!/usr/bin/env bash
set -euo pipefail

SOC_VERSION="Ascend910B3"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
PYTHON_BIN="${PYTHON:-python3}"
BUILD_DIR="build"
RUN_TEST=1

while getopts "v:a:p:b:t" opt; do
  case ${opt} in
    v) SOC_VERSION=${OPTARG} ;;
    a) ASCEND_CANN_PACKAGE_PATH=${OPTARG} ;;
    p) PYTHON_BIN=${OPTARG} ;;
    b) BUILD_DIR=${OPTARG} ;;
    t) RUN_TEST=0 ;;
    *) echo "Usage: bash run.sh [-v SOC_VERSION] [-a ASCEND_CANN_PACKAGE_PATH] [-p python] [-b build_dir] [-t skip_test]"; exit 1 ;;
  esac
done

if [ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

${PYTHON_BIN} - <<'PY'
try:
    import pybind11  # noqa: F401
except Exception:
    raise SystemExit("pybind11 is required: pip install pybind11")
try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
except Exception as e:
    raise SystemExit(f"torch/torch_npu import failed: {e}")
PY

cmake -S . -B "${BUILD_DIR}" \
  -DRUN_MODE=npu \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DPython3_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j

if [ "${RUN_TEST}" -eq 1 ]; then
  LD_LIBRARY_PATH="${PWD}/${BUILD_DIR}/lib:${PWD}/${BUILD_DIR}:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="${PWD}/${BUILD_DIR}:${PYTHONPATH:-}" \
    ${PYTHON_BIN} scripts/test_lookup_update.py
fi
