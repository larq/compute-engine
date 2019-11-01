#!/bin/bash
set -e

# Use --fullbuild to also build the pip package

# Use --benchmark to compile with gemmlowp profiling enabled
# Note: if you have previously build the library without benchmarking
# then you should pass --cleanbuild to do a complete rebuild

fullbuild=0
benchmark=0
cleanbuild=0

for arg in "$@"
do
case $arg in
    --fullbuild)
	fullbuild=1
    shift
    ;;
    --benchmark)
	benchmark=1
    shift
    ;;
    --cleanbuild)
    cleanbuild=1
    shift
    ;;
    *)
    ;;
esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/ext/tensorflow"

# Try to figure out the host system
HOST_OS="unknown"
UNAME="$(uname -s)"
if [ "$UNAME" = "Linux" ] ; then
    HOST_OS="linux"
elif [ "$UNAME" = "Darwin" ] ; then
    HOST_OS="osx"
fi
HOST_ARCH="$(if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)"

export EXTRA_CXXFLAGS="-DTFLITE_WITH_RUY"
if [ "$benchmark" == "1" ]; then
	export EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS} -DGEMMLOWP_PROFILING"
fi

if [ "$cleanbuild" == "1" ]; then
    rm -rf "${TF_DIR}/tensorflow/lite/tools/make/gen"
fi

# Check if dependencies need to be downloaded
if [ ! -d "${TF_DIR}/tensorflow/lite/tools/make/downloads" ]; then
    ${TF_DIR}/tensorflow/lite/tools/make/download_dependencies.sh
fi
# Build tflite (will automatically skip when up-to-date
${TF_DIR}/tensorflow/lite/tools/make/build_lib.sh
# Build our kernels
make -j 8 BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
# Rebuild tflite binaries such as benchmarking libs to include our ops
${TF_DIR}/tensorflow/lite/tools/make/build_lib.sh

# Optionally build the pip package
if [ "$fullbuild" == "1" ]; then
    ${TF_DIR}/tensorflow/lite/tools/pip_package/build_pip_package.sh
fi


#
# Everything below is for cross-compilation only.
# By default it won't run, except if that cross-compiled target already existed,
# in which case it will add our kernels to it.
#


# Check if we need to cross-compile to raspberry pi
if [ -f "${TF_DIR}/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a" ]; then
    make -j 8 TARGET=rpi -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
    ${TF_DIR}/tensorflow/lite/tools/make/build_rpi_lib.sh
fi

# Check if we need to compile for iOS
IOS_BUILT=false
IOS_ARCHS="x86_64 armv7 armv7s arm64"
for arch in $BUILD_ARCHS
do
    if [ -f "${TF_DIR}/tensorflow/lite/tools/make/gen/ios_${arch}/lib/libtensorflow-lite.a" ]; then
        IOS_BUILT=true
        make -j 8 TARGET=ios TARGET_ARCH=${arch} -C "${ROOT_DIR}" -f larq_compute_engine/tflite/build/Makefile
    fi
done
if [ "$IOS_BUILT" = true ] ; then
    ${TF_DIR}/tensorflow/lite/tools/make/build_ios_universal_lib.sh
fi
