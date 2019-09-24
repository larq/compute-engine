#!/bin/bash
set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORFLOW_DIR="${SCRIPT_DIR}/../tensorflow"

make -j 4 TARGET=rpi -C "${TENSORFLOW_DIR}" -f ../tflite/Makefile

