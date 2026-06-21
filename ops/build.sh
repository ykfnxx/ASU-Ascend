#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash build.sh lookup_aiv [SOC_VERSION]
  bash build.sh maintain_aicpu [SOC_VERSION]
  bash build.sh maintain_aicpu_msopgen
  bash build.sh all [SOC_VERSION]

Environment:
  ASCEND_CANN_PACKAGE_PATH  CANN ascend-toolkit path for lookup_aiv
  ASCEND_INSTALL_PATH       Fallback Ascend install prefix
  NPU_ARCH                  ASC host npu arch, default: dav-2201
  MSOPGEN_BIN               Optional path to msopgen
  BUILD_ROOT                Build directory root, default: build
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

OP="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC_VERSION="${2:-${SOC_VERSION:-Ascend910B3}}"
NPU_ARCH="${NPU_ARCH:-dav-2201}"
BUILD_ROOT="${BUILD_ROOT:-build}"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"

build_lookup_aiv() {
  local op="lookup_aiv"
  local build_dir
  if [[ "${BUILD_ROOT}" = /* ]]; then
    build_dir="${BUILD_ROOT}/${op}"
  else
    build_dir="${SCRIPT_DIR}/${BUILD_ROOT}/${op}"
  fi

  if [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
  fi

  cmake -S "${SCRIPT_DIR}" -B "${build_dir}" \
    -DASU_HBM_INDEX_OP="${op}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build "${build_dir}" --target "asu_hbm_index_${op}" -j
}

find_msopgen() {
  if [[ -n "${MSOPGEN_BIN:-}" ]]; then
    echo "${MSOPGEN_BIN}"
    return
  fi
  if command -v msopgen >/dev/null 2>&1; then
    command -v msopgen
    return
  fi
  if [[ -x "${ASCEND_CANN_PACKAGE_PATH}/python/site-packages/bin/msopgen" ]]; then
    echo "${ASCEND_CANN_PACKAGE_PATH}/python/site-packages/bin/msopgen"
    return
  fi
  if [[ -x "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel/msopgen" ]]; then
    echo "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel/msopgen"
    return
  fi
  return 1
}

build_maintain_aicpu() {
  local op="maintain_aicpu"
  local build_dir
  if [[ "${BUILD_ROOT}" = /* ]]; then
    build_dir="${BUILD_ROOT}/${op}"
  else
    build_dir="${SCRIPT_DIR}/${BUILD_ROOT}/${op}"
  fi

  if [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
  fi

  cmake -S "${SCRIPT_DIR}" -B "${build_dir}" \
    -DASU_HBM_INDEX_OP="${op}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DNPU_ARCH="${NPU_ARCH}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build "${build_dir}" --target "asu_hbm_index_${op}" -j
}

build_maintain_aicpu_msopgen() {
  local op="maintain_aicpu_msopgen"
  local build_dir
  if [[ "${BUILD_ROOT}" = /* ]]; then
    build_dir="${BUILD_ROOT}/${op}"
  else
    build_dir="${SCRIPT_DIR}/${BUILD_ROOT}/${op}"
  fi

  if [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
  fi

  local msopgen_bin
  if ! msopgen_bin="$(find_msopgen)"; then
    echo "Error: msopgen was not found. Set MSOPGEN_BIN or source the Ascend CANN environment." >&2
    exit 1
  fi

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}"

  "${msopgen_bin}" gen \
    -i "${SCRIPT_DIR}/ir/asu_hbm_index_maintain_aicpu.json" \
    -f tf \
    -c aicpu \
    -out "${build_dir}"

  mkdir -p "${build_dir}/asu_source"
  cp "${SCRIPT_DIR}/asu_hbm_index_maintain_aicpu.cpp" "${build_dir}/asu_source/"
  cp "${SCRIPT_DIR}/asu_hbm_index_maintain_aicpu_kernel.aicpu" "${build_dir}/asu_source/"

  echo "AICPU msopgen project generated at: ${build_dir}"
  echo "Port the asu_source/asu_hbm_index_maintain_aicpu_kernel.aicpu logic into generated cpukernel/impl/*_kernels.cc, then run the generated build script."
}

case "${OP}" in
  lookup_aiv|maintain_aicpu|maintain_aicpu_msopgen)
    if [[ "${OP}" == "lookup_aiv" ]]; then
      build_lookup_aiv
    elif [[ "${OP}" == "maintain_aicpu" ]]; then
      build_maintain_aicpu
    else
      build_maintain_aicpu_msopgen
    fi
    ;;
  all)
    build_lookup_aiv
    build_maintain_aicpu
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
