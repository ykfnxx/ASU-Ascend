#!/usr/bin/env bash
set -euo pipefail

# Minimal convenience wrapper for a CANN custom-op build environment.
# It intentionally does not reuse the standalone simu demo build.

SOC_VERSION="${1:-ascend910b}"
BUILD_DIR="${BUILD_DIR:-build}"

cmake -S . -B "${BUILD_DIR}" \
  -DASCEND_OP_NAME=asu_resolve_kv_slots \
  -DASCEND_COMPUTE_UNIT="${SOC_VERSION}"

cmake --build "${BUILD_DIR}" -j
