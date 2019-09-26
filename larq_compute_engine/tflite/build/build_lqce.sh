#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."

if [ -f "${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a" ]; then
    make -j 8 BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
    # Also re-make the tf-lite examples so that they use the modified library
    ${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/build_lib.sh
else
    echo "linux_x86_64 target not found. Please build tensorflow lite first using build_lib.sh"
fi

# Check if we need to cross-compile to raspberry pi
if [ -f "${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a" ]; then
    make -j 8 TARGET=rpi -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
    ${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh
fi

# Check if we need to compile for iOS
IOS_BUILT=false
IOS_ARCHS="x86_64 armv7 armv7s arm64"
for arch in $BUILD_ARCHS
do
    if [ -f "${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/gen/ios_${arch}/lib/libtensorflow-lite.a" ]; then
        IOS_BUILT=true
        make -j 8 TARGET=ios TARGET_ARCH=${arch} -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
    fi
done
if [ "$IOS_BUILT" = true ] ; then
    ${ROOT_DIR}/tensorflow/tensorflow/lite/tools/make/build_ios_universal_lib.sh
fi
