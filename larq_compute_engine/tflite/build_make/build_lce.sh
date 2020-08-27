#!/bin/bash
set -e

usage()
{
    echo "Usage: build_lqce.sh [--native] [--rpi] [--ios] [--aarch64] [--benchmark] [--clean]

--native        Build for the host platform
--rpi           Compile for Raspberry Pi (32-bit armv7)
--ios           Compile for iOS
--aarch64       Compile for Aarch64 (e.g. 64-bit Raspberry Pi)

When building on a Raspberry Pi, it is advised to use the --rpi or --aarch64 options instead of --native, in order to set the correct compiler optimization flags.

For cross-compiling, the relevant toolchains have to be installed using the systems package manager.
The --rpi option requires the arm-linux-gnueabihf toolchain.
The --aarch64 option requires the aarch64-linux-gnu toolchain.
The --ios option requires the iOS SDK.

--benchmark     Compile with RUY profiling enabled
--clean         Delete intermediate build files

If doing a benchmark build when you have previously built without --benchmark
then you should pass --clean to do a complete rebuild."
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
TF_GEN_DIR="${TF_DIR}/tensorflow/lite/tools/make/gen"
export LCE_GEN_DIR="${ROOT_DIR}/gen" # export to pass it to the Makefile

# number of hyper threads
NUM_HYPERTHREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)

export BUILD_WITH_RUY=true
if [ "$benchmark" == "1" ]; then
    export BUILD_WITH_RUY_PROFILER=true
fi

if [ "$clean" == "1" ]; then
    echo " --> clean"
    rm -rf ${TF_GEN_DIR}
    rm -rf ${LCE_GEN_DIR}
fi

# Check if dependencies need to be downloaded
if [ ! -d "${TF_DIR}/tensorflow/lite/tools/make/downloads" ]; then
    ${TF_DIR}/tensorflow/lite/tools/make/download_dependencies.sh
fi

if [ "$native" == "1" ]; then
    echo " --> native build"
    # Build the tflite lib (will automatically skip when up-to-date
    # this line is taken from "tensorflow/lite/tools/make/build_lib.sh"
    make -j ${NUM_HYPERTHREADS} BUILD_WITH_NNAPI=false -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels and benchmark binary
    make -j ${NUM_HYPERTHREADS} BUILD_WITH_NNAPI=false -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
fi

if [ "$rpi" == "1" ]; then
    echo " --> rpi build"
    # Stored in gen/rpi_armv7l
    # This line is taken form "tensorflow/lite/tools/make/build_rpi_lib.sh"
    make -j ${NUM_HYPERTHREADS} TARGET=rpi -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels and benchmark binary
    make -j ${NUM_HYPERTHREADS} TARGET=rpi -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
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
fi

if [ "$aarch64" == "1" ]; then
    echo " --> aarch64 build"
    # Stored in gen/aarch64_armv8-a
    # This line is taken from "tensorflow/lite/tools/make/build_aarch64_lib.sh"
    make -j ${NUM_HYPERTHREADS} TARGET=aarch64 -C "${TF_DIR}" -f tensorflow/lite/tools/make/Makefile
    # Build compute-engine kernels and benchmark binary
    make -j ${NUM_HYPERTHREADS} TARGET=aarch64 -C "${ROOT_DIR}" -f ${LCE_MAKEFILE}
fi
