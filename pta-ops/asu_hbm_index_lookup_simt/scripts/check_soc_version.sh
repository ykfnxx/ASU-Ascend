#!/usr/bin/env bash
set -euo pipefail

DEVICE_ID="0"
BOARD_INFO_FILE=""
VALUE_ONLY=0
RUNTIME_SOC_OVERRIDE=""
PYTHON_BIN_ARG="${PYTHON_BIN:-python3}"
CANN_PATH_ARG="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"

usage() {
  cat <<'EOF'
Query the runtime SOC_VERSION without guessing its format.

Usage:
  scripts/check_soc_version.sh [options]

Options:
  --device ID             NPU device id passed to npu-smi (default: 0)
  --cann-path PATH        CANN ascend-toolkit path
  --python PATH           Python with torch and torch-npu (default: python3)
  --board-info-file PATH  Parse saved npu-smi board output instead of hardware
  --runtime-soc SOC       Use a supplied runtime value (offline/testing)
  --value-only            Print only the SOC_VERSION value
  -h, --help              Show this help
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
    --device)
      require_option_value "$@"
      DEVICE_ID="$2"
      shift 2
      ;;
    --cann-path)
      require_option_value "$@"
      CANN_PATH_ARG="$2"
      shift 2
      ;;
    --python)
      require_option_value "$@"
      PYTHON_BIN_ARG="$2"
      shift 2
      ;;
    --board-info-file)
      require_option_value "$@"
      BOARD_INFO_FILE="$2"
      shift 2
      ;;
    --runtime-soc)
      require_option_value "$@"
      RUNTIME_SOC_OVERRIDE="$2"
      shift 2
      ;;
    --value-only)
      VALUE_ONLY=1
      shift
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

if ! [[ "${DEVICE_ID}" =~ ^[0-9]+$ ]]; then
  echo "--device must be a non-negative integer; got ${DEVICE_ID}" >&2
  exit 2
fi

if [[ -f "${CANN_PATH_ARG}/set_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CANN_PATH_ARG}/set_env.sh"
  set -u
fi

if [[ -n "${RUNTIME_SOC_OVERRIDE}" ]]; then
  DETECTED_SOC_VERSION="${RUNTIME_SOC_OVERRIDE}"
else
  if ! command -v "${PYTHON_BIN_ARG}" >/dev/null 2>&1; then
    echo "Python interpreter is not available: ${PYTHON_BIN_ARG}" >&2
    exit 2
  fi
  if ! DETECTED_SOC_VERSION="$(
    "${PYTHON_BIN_ARG}" - "${DEVICE_ID}" <<'PY'
import sys

import torch
import torch_npu  # noqa: F401

device_id = int(sys.argv[1])
torch.npu.set_device(device_id)
soc_version = torch.npu.get_device_name(device_id)
if not soc_version:
    raise RuntimeError("torch.npu.get_device_name returned an empty value")
print(soc_version)
PY
  )"; then
    echo "failed to query SOC_VERSION through the Ascend runtime" >&2
    echo "check that torch-npu and the CANN runtime are available" >&2
    exit 2
  fi
fi

if [[ -z "${DETECTED_SOC_VERSION}" ||
      "${DETECTED_SOC_VERSION}" == *$'\n'* ]]; then
  echo "runtime returned an invalid SOC_VERSION value" >&2
  exit 2
fi

if ((VALUE_ONLY)); then
  printf '%s\n' "${DETECTED_SOC_VERSION}"
  exit 0
fi

BOARD_INFO=""
if [[ -n "${BOARD_INFO_FILE}" ]]; then
  if [[ ! -f "${BOARD_INFO_FILE}" ]]; then
    echo "board info file does not exist: ${BOARD_INFO_FILE}" >&2
    exit 2
  fi
  BOARD_INFO="$(<"${BOARD_INFO_FILE}")"
elif command -v npu-smi >/dev/null 2>&1; then
  if ! BOARD_INFO="$(npu-smi info -t board -i "${DEVICE_ID}" 2>&1)"; then
    echo "failed to query board info for NPU device ${DEVICE_ID}" >&2
    printf '%s\n' "${BOARD_INFO}" >&2
    exit 2
  fi
else
  echo "warning: npu-smi is unavailable; skipping board diagnostics" >&2
fi

extract_field() {
  local field_name="$1"
  awk -F: -v field_name="${field_name}" '
    {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == field_name) {
        value = substr($0, index($0, ":") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
  ' <<<"${BOARD_INFO}"
}

printf 'DEVICE_ID=%s\n' "${DEVICE_ID}"
printf 'SOC_VERSION=%s\n' "${DETECTED_SOC_VERSION}"
if [[ -n "${BOARD_INFO}" ]]; then
  CHIP_NAME="$(extract_field "Chip Name")"
  NPU_NAME="$(extract_field "NPU Name")"
  printf 'BOARD_CHIP_NAME=%s\n' "${CHIP_NAME:-unavailable}"
  printf 'BOARD_NPU_NAME=%s\n' "${NPU_NAME:-unavailable}"
fi
printf 'CANN_PATH=%s\n' "${CANN_PATH_ARG}"

if [[ -f "${CANN_PATH_ARG}/version.cfg" ]]; then
  printf 'CANN_VERSION_FILE=%s\n' "${CANN_PATH_ARG}/version.cfg"
  grep -E -i '(^|_)(version|version_dir)=' \
    "${CANN_PATH_ARG}/version.cfg" | head -n 4 || true
else
  echo "warning: CANN version.cfg was not found" >&2
fi

CMAKE_SEARCH_DIRS=()
for candidate in \
  "${CANN_PATH_ARG}/tools/tikcpp/ascendc_kernel_cmake" \
  "${CANN_PATH_ARG}/compiler/tikcpp/ascendc_kernel_cmake"; do
  if [[ -d "${candidate}" ]]; then
    CMAKE_SEARCH_DIRS+=("${candidate}")
  fi
done

if ((${#CMAKE_SEARCH_DIRS[@]} == 0)); then
  echo "warning: ascendc_kernel_cmake was not found under the CANN path" >&2
elif grep -R -I -F -m 1 -- "${DETECTED_SOC_VERSION}" \
  "${CMAKE_SEARCH_DIRS[@]}" >/dev/null 2>&1; then
  echo "CANN_SOC_REFERENCE=found"
else
  echo "CANN_SOC_REFERENCE=not-found"
  echo "warning: the exact SOC string was not found in AscendC CMake files;" >&2
  echo "         the compiler configure step remains the definitive check" >&2
fi
