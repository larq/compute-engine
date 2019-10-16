#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/ext/tensorflow"

# Take the first argument as modelfile
# but use a default when its not given.
MODELFILE=${1:-benchmarknet.tflite}
OUTPUTFILE=${2:-benchmarknet_results.txt}

if [ ! -f "${MODELFILE}" ]; then
    echo "File not found: ${MODELFILE}."
    echo "Usage: $0 model_filename ouput_filename"

    if [ "$MODELFILE" = "benchmarknet.tflite" ]; then
        read -p "Do you want to generate benchmarknet.tflite? Y to generate and then benchmark, N to quit. [y/N] " INPUT
        case $INPUT in
            [Yy]* ) echo "Generating tflite model."; GENERATE=1;;
            [Nn]* ) echo "Not generating tflite model."; GENERATE=0;;
            "" ) echo "Not generating tflite model."; GENERATE=0;;
            * ) echo "Invalid selection: " $INPUT;;
        esac
        if [ "$GENERATE" = "1" ]; then
            python3 benchmarknet.py
        else
            exit 1
        fi
    else
        exit 1
    fi
fi

# Try to figure out the host system
HOST_OS="unknown"
UNAME="$(uname -s)"
if [ "$UNAME" = "Linux" ] ; then
    HOST_OS="linux"
elif [ "$UNAME" = "Darwin" ] ; then
    HOST_OS="osx"
fi
HOST_ARCH="$(if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)"

BENCHMARK_MODEL="${TF_DIR}/tensorflow/lite/tools/make/gen/${HOST_OS}_${HOST_ARCH}/bin/benchmark_model"
if [ -f "${BENCHMARK_MODEL}" ]; then
    ${BENCHMARK_MODEL} --graph="${MODELFILE}" --enable_op_profiling=true | tee ${OUTPUTFILE}
else
    echo "${HOST_OS}_${HOST_ARCH} benchmark binary not found. Please build the larq compute engine (tf lite part) first."
    exit 1
fi
