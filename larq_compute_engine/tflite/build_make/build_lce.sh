#!/bin/bash
set -e

usage()
{
    echo "Usage: build_lqce.sh [--native] [--rpi] [--ios] [--aarch64] [--benchmark] [--clean] [--pip]

--native        Build for the host platform
--rpi           Cross-compile for Raspberry Pi (32-bit armv7)
--ios           Cross-compile for iOS
--aarch64       Cross-compile for Aarch64 (e.g. 64-bit Raspberry Pi)

When building on a 32-bit Raspberry Pi, it is advised to use the --rpi option in order to set the correct compiler optimization flags.

For cross-compiling, the relevant toolchains have to be installed using the systems package manager.
The --rpi option requires the arm-linux-gnueabihf toolchain.
The --aarch64 option requires the aarch64-linux-gnu toolchain.
The --ios option requires the iOS SDK.

--benchmark     Compile with gemmlowp profiling enabled
--clean         Delete intermediate build files
--pip           Also build the pip package

If doing a benchmark build when you have previously built without --benchmark
then you should pass --clean to do a complete rebuild.

Building the pip package is only available for the --native target."
}


if [[ $# -eq 0 ]] ; then
    usage
    exit 0
fi

native=0
rpi=0
ios=0
aarch64=0
benchmark=0
clean=0
pip=0

while [ "$1" != "" ]; do
    case $1 in
        -n | --native)
	    native=1
            ;;
        --rpi)
	    rpi=1
            ;;
        --ios)
	    ios=1
            ;;
        --aarch64)
	    aarch64=1
            ;;
        -b | --benchmark)
	    benchmark=1
            ;;
        -c | --clean)
            clean=1
            ;;
        --pip)
	    pip=1
            ;;
        -h | --help )
            usage
            exit
            ;;
        * )
            usage
            exit 1
            ;;
    esac
    shift
done


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/third_party/tensorflow"
LCE_MAKEFILE="larq_compute_engine/tflite/build_make/Makefile"

# number of hyper threads
NUM_HYPERTHREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)

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

if [ "$clean" == "1" ]; then
    echo " --> clean"
    rm -rf "${TF_DIR}/tensorflow/lite/tools/make/gen"
fi

# Check if dependencies need to be downloaded
if [ ! -d "${TF_DIR}/tensorflow/lite/tools/make/downloads" ]; then
    ${TF_DIR}/tensorflow/lite/tools/make/download_dependencies.sh
fi

if [ "$native" == "1" ]; then
    echo " --> native build"
    # Build tflite (will automatically skip when up-to-date
    # this line is taken from "tensorflow/lite/tools/make/build_lib.sh"
    make -j ${NUM_HYPERTHREADS} BUILD_WITH_NNAPI=false -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels
    make -j ${NUM_HYPERTHREADS} BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
    # Rebuild tflite binaries such as benchmarking libs to include our ops
    ${TF_DIR}/tensorflow/lite/tools/make/build_lib.sh

    # Optionally build the pip package
    if [ "$pip" == "1" ]; then
        ${TF_DIR}/tensorflow/lite/tools/pip_package/build_pip_package.sh
    fi
fi

if [ "$rpi" == "1" ]; then
    echo " --> rpi build"
    # Stored in gen/rpi_armv7l
    # This line is taken form "tensorflow/lite/tools/make/build_rpi_lib.sh"
    make -j ${NUM_HYPERTHREADS} TARGET=rpi -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels
    make -j ${NUM_HYPERTHREADS} TARGET=rpi -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
    ${TF_DIR}/tensorflow/lite/tools/make/build_rpi_lib.sh

    # Optionally build the pip package
    if [ "$pip" == "1" ]; then
        # This will only work if we are running this natively on a raspberry pi
        if [[ $HOST_ARCH == armv7* ]]; then
            ${TF_DIR}/tensorflow/lite/tools/pip_package/build_pip_package.sh
        fi
    fi
fi

if [ "$ios" == "1" ]; then
    echo " --> ios build"
    profiling_opt=""
    if [ "$benchmark" == "1" ]; then
        profiling_opt="-p"
    fi
    ${TF_DIR}/tensorflow/lite/tools/make/build_ios_universal_lib.sh $profiling_opt
    IOS_ARCHS="x86_64 armv7 armv7s arm64"
    for arch in $BUILD_ARCHS
    do
        # Stored in gen/ios_$arch
        make -j ${NUM_HYPERTHREADS} TARGET=ios TARGET_ARCH=${arch} -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
    done
    ${TF_DIR}/tensorflow/lite/tools/make/build_ios_universal_lib.sh $profiling_opt
fi

if [ "$aarch64" == "1" ]; then
    echo " --> aarch64 build"
    # Stored in gen/aarch64_armv8-a
    # This line is taken from "tensorflow/lite/tools/make/build_aarch64_lib.sh"
    make -j ${NUM_HYPERTHREADS} TARGET=aarch64 -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels
    make -j ${NUM_HYPERTHREADS} TARGET=aarch64 -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
    ${TF_DIR}/tensorflow/lite/tools/make/build_aarch64_lib.sh
fi
