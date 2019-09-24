#!/bin/bash -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORFLOW_DIR="${SCRIPT_DIR}/../tensorflow"

CC_PREFIX=aarch64-linux-gnu- make -j 3 -C "${TENSORFLOW_DIR}" -f ../tflite/Makefile TARGET=generic-aarch64 TARGET_ARCH=armv8-a
