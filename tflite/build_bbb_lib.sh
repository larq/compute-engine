#!/bin/bash -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORFLOW_DIR="${SCRIPT_DIR}/../tensorflow"

CC_PREFIX=arm-linux-gnueabihf- make -j 3 -C "${TENSORFLOW_DIR}" -f ../tflite/Makefile TARGET=bbb TARGET_ARCH=armv7l
