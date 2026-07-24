#!/usr/bin/env bash
set -euo pipefail

DEVICE_ID="0"
BOARD_INFO_FILE=""
VALUE_ONLY=0
CANN_PATH_ARG="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"

usage() {
  cat <<'EOF'
Detect the full SOC_VERSION for Ascend 950PR/950DT.

SOC_VERSION is composed as:
  <Chip Name>_<NPU Name>

Usage:
  scripts/check_soc_version.sh [options]

Options:
  --device ID             NPU device id passed to npu-smi (default: 0)
  --cann-path PATH        CANN ascend-toolkit path
  --board-info-file PATH  Parse saved npu-smi board output instead of hardware
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
    --board-info-file)
      require_option_value "$@"
      BOARD_INFO_FILE="$2"
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

if [[ -n "${BOARD_INFO_FILE}" ]]; then
  if [[ ! -f "${BOARD_INFO_FILE}" ]]; then
    echo "board info file does not exist: ${BOARD_INFO_FILE}" >&2
    exit 2
  fi
  BOARD_INFO="$(<"${BOARD_INFO_FILE}")"
else
  if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi is not available; run on the Ascend host" >&2
    exit 2
  fi
  if ! BOARD_INFO="$(npu-smi info -t board -i "${DEVICE_ID}" 2>&1)"; then
    echo "failed to query board info for NPU device ${DEVICE_ID}" >&2
    printf '%s\n' "${BOARD_INFO}" >&2
    exit 2
  fi
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

CHIP_NAME="$(extract_field "Chip Name")"
NPU_NAME="$(extract_field "NPU Name")"

if [[ -z "${CHIP_NAME}" || -z "${NPU_NAME}" ]]; then
  echo "could not parse both 'Chip Name' and 'NPU Name' from board info" >&2
  printf '%s\n' "${BOARD_INFO}" >&2
  exit 2
fi
if [[ ! "${CHIP_NAME}" =~ ^Ascend950(PR|DT)$ ]]; then
  echo "unsupported chip for this operator: ${CHIP_NAME}" >&2
  exit 2
fi
if [[ ! "${NPU_NAME}" =~ ^[[:alnum:]_.-]+$ ]]; then
  echo "NPU Name contains unsupported characters: ${NPU_NAME}" >&2
  exit 2
fi

DETECTED_SOC_VERSION="${CHIP_NAME}_${NPU_NAME}"
if ((VALUE_ONLY)); then
  printf '%s\n' "${DETECTED_SOC_VERSION}"
  exit 0
fi

printf 'DEVICE_ID=%s\n' "${DEVICE_ID}"
printf 'CHIP_NAME=%s\n' "${CHIP_NAME}"
printf 'NPU_NAME=%s\n' "${NPU_NAME}"
printf 'SOC_VERSION=%s\n' "${DETECTED_SOC_VERSION}"
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
