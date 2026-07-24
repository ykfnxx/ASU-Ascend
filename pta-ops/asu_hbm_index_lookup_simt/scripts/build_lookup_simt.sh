#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR_ARG="build"
CANN_PATH_ARG="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"
SOC_VERSION_ARG="${SOC_VERSION:-}"
JOBS_ARG="$(nproc)"

usage() {
  cat <<'EOF'
Build asu_hbm_index_lookup_simt for Ascend 950.

Usage:
  scripts/build_lookup_simt.sh [options]

Options:
  --build-dir PATH  CMake build directory (default: build under package)
  --cann-path PATH  CANN ascend-toolkit path
  --soc-version SOC Exact SOC_VERSION (required unless set in environment)
  --jobs N          Parallel build jobs (default: nproc)
  -h, --help        Show this help
EOF
}

require_option_value() {
  if (($# < 2)); then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --build-dir)
      require_option_value "$@"
      BUILD_DIR_ARG="$2"
      shift 2
      ;;
    --cann-path)
      require_option_value "$@"
      CANN_PATH_ARG="$2"
      shift 2
      ;;
    --soc-version)
      require_option_value "$@"
      SOC_VERSION_ARG="$2"
      shift 2
      ;;
    --jobs)
      require_option_value "$@"
      JOBS_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${BUILD_DIR_ARG}" = /* ]]; then
  BUILD_DIR="${BUILD_DIR_ARG}"
else
  BUILD_DIR="${PKG_DIR}/${BUILD_DIR_ARG}"
fi

if ! [[ "${JOBS_ARG}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer; got ${JOBS_ARG}" >&2
  exit 2
fi
if [[ -z "${SOC_VERSION_ARG}" ]]; then
  echo "--soc-version is required when SOC_VERSION is not set" >&2
  exit 2
fi
if [[ ! -d "${CANN_PATH_ARG}" ]]; then
  echo "CANN path does not exist: ${CANN_PATH_ARG}" >&2
  exit 2
fi
if [[ -f "${CANN_PATH_ARG}/set_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CANN_PATH_ARG}/set_env.sh"
  set -u
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is not available in PATH" >&2
  exit 2
fi
echo "SOC_VERSION=${SOC_VERSION_ARG}"
echo "ASCEND_CANN_PACKAGE_PATH=${CANN_PATH_ARG}"
echo "BUILD_DIR=${BUILD_DIR}"

cmake -S "${PKG_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC_VERSION_ARG}" \
  -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH_ARG}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS_ARG}"

LIBRARY_PATH="${BUILD_DIR}/lib/libasu_hbm_index_lookup_simt_kernel.so"
if [[ ! -f "${LIBRARY_PATH}" ]]; then
  LIBRARY_PATH="$(
    find "${BUILD_DIR}" -type f \
      -name 'libasu_hbm_index_lookup_simt_kernel.so' -print -quit
  )"
fi
if [[ -z "${LIBRARY_PATH}" || ! -f "${LIBRARY_PATH}" ]]; then
  echo "build finished but the SIMT kernel library was not found under ${BUILD_DIR}" >&2
  exit 1
fi
if command -v nm >/dev/null 2>&1 && ! nm -D --defined-only "${LIBRARY_PATH}" |
  awk '$NF == "asu_hbm_index_lookup_simt_do" { found = 1 } END { exit !found }'; then
  echo "kernel library does not export asu_hbm_index_lookup_simt_do: ${LIBRARY_PATH}" >&2
  exit 1
fi
echo "built library: ${LIBRARY_PATH}"
