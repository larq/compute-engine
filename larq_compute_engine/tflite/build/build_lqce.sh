#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/ext/tensorflow"
LQCE_MAKEFILE="larq_compute_engine/tflite/build/Makefile" #relative to ROOT_DIR

# Try to figure out the host system
HOST_OS="unknown"
UNAME="$(uname -s)"
if [ "$UNAME" = "Linux" ] ; then
    HOST_OS="linux"
elif [ "$UNAME" = "Darwin" ] ; then
    HOST_OS="osx"
fi
HOST_ARCH="$(if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)"

LIB_DIR="${TF_DIR}/tensorflow/lite/tools/make/gen/${HOST_OS}_${HOST_ARCH}/lib"
TFLITE_LIB="${LIB_DIR}/libtensorflow-lite.a"
TFLITE_BENCH_LIB="${LIB_DIR}/benchmark-lib.a"
PREBUILT_LIB="${ROOT_DIR}/ext/${HOST_OS}_${HOST_ARCH}/libtensorflow-lite.a"
PREBUILT_BENCH_LIB="${ROOT_DIR}/ext/${HOST_OS}_${HOST_ARCH}/benchmark-lib.a"

# Check if a pre-built libtensorflow-lite.a is available
if [[ ! -f ${TFLITE_LIB} || ! -f ${TFLITE_BENCH_LIB} ]]; then
    if [[ -f "${PREBUILT_LIB}" && -f "${PREBUILT_BENCH_LIB}" ]]; then
        echo "Prebuilt tflite libraries available."
	mkdir -p ${LIB_DIR}
        cp ${PREBUILT_LIB} ${TFLITE_LIB}
        cp ${PREBUILT_BENCH_LIB} ${TFLITE_BENCH_LIB}
    else
        echo "${HOST_OS}_${HOST_ARCH} target not found and no prebuilt library available for this target."
        echo "Please build tensorflow lite first using build_lib.sh"
    fi
fi
if [ -f ${TFLITE_LIB} ]; then
    # Add our additions to the tflite library
    make -j 8 BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f ${LQCE_MAKEFILE}
    # Also re-make the tf-lite C++ examples so that they use the modified library
    # We use some special flags to make sure it will not recompile the library
    # Somehow, realpath is required for --assume-old and --assume new in make
    TFLITE_LIB="$(realpath ${TFLITE_LIB})"
    TFLITE_BENCH_LIB="$(realpath ${TFLITE_BENCH_LIB})"
    make -j 8 BUILD_WITH_NNAPI=false --assume-new=${TFLITE_LIB} --assume-new=${TFLITE_BENCH_LIB} --assume-old=${TFLITE_LIB} --assume-old=${TFLITE_BENCH_LIB} -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Relevant make options explained
    # -B --always-make
    #   Make target, such as `minimal`, even if its up to date to enforce use of our library
    # -o --old-file --assume-old=libtensorflow.a
    #   Mark `libtensorflow-lite.a` as not needing to be rebuilt even
    #   if the .o files it depends on are newer. Good!
    #   This will also stop it from remaking `minimal`. Bad!
    #   However, we can combine it with -B or with -W which will cause it to
    #   remake `minimal` without remaking the library.
    # -t --touch
    #   Touch access times on files to pretend commands were done
    #   in order to fool future invocations of make.
    #   Tried it, but if the .o files do not yet exist it seems they wont be created by this
    # -W --what-if --new-file --assume-new
    #   Seems to be the opposite of what we want.
    #   However, we can combine it with -o and it will rebuild
    #   dependencies.
fi

# Check if we need to cross-compile to raspberry pi
if [ -f "${TF_DIR}/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a" ]; then
    make -j 8 TARGET=rpi -C "${ROOT_DIR}" -f ${LQCE_MAKEFILE}
    ${TF_DIR}/tensorflow/lite/tools/make/build_rpi_lib.sh
fi

# Check if we need to compile for iOS
IOS_BUILT=false
IOS_ARCHS="x86_64 armv7 armv7s arm64"
for arch in $BUILD_ARCHS
do
    if [ -f "${TF_DIR}/tensorflow/lite/tools/make/gen/ios_${arch}/lib/libtensorflow-lite.a" ]; then
        IOS_BUILT=true
        make -j 8 TARGET=ios TARGET_ARCH=${arch} -C "${ROOT_DIR}" -f ${LQCE_MAKEFILE}
    fi
done
if [ "$IOS_BUILT" = true ] ; then
    ${TF_DIR}/tensorflow/lite/tools/make/build_ios_universal_lib.sh
fi
