#!/bin/bash
set -x
set -e

# TODO: Check if other build script has been run first!

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."

make -j 4 BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile

