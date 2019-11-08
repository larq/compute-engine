#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/ext/tensorflow"

if [ "$1" != "" ]; then
    if [ ! -f "$1" ]; then
        echo "File not found: $1."
        echo "Usage:"
        echo "    $0 model_filename     Benchmark one model"
        echo "    $0                    Benchmark all models in benchmarking_models/"
        exit 1
    fi
    ALLMODELS="$1"
else
    if ! compgen -G "benchmarking_models/*.tflite*" > /dev/null; then
        echo "No models found in benchmarking_models/"
        read -p "Do you want to generate benchmarking tflite models? Y to generate and then benchmark, N to quit. [y/N] " INPUT
        case $INPUT in
            [Yy]* ) echo "Generating tflite models."; GENERATE=1;;
            [Nn]* ) echo "Not generating tflite models."; GENERATE=0;;
            "" ) echo "Not generating tflite models."; GENERATE=0;;
            * ) echo "Invalid selection: " $INPUT;;
        esac
        if [ "$GENERATE" = "1" ]; then
            python3 benchmarknet.py
        else
            exit 1
        fi
    fi
    ALLMODELS=$(ls benchmarking_models/*.tflite)
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

if [[ $HOST_ARCH == armv7* ]]; then
    # Probably this is a 32-bit Raspberry Pi
    # In this case, also check the rpi_armv7l folder
    BENCHMARK_MODEL="${TF_DIR}/tensorflow/lite/tools/make/gen/rpi_armv7l/bin/benchmark_model"
    if [ ! -f "${BENCHMARK_MODEL}" ]; then
        echo "rpi_armv7l benchmark binary not found. Trying other binary as well."
        unset BENCHMARK_MODEL
    fi
fi

# The := will not change BENCHMARK_MODEL when it was already set
# So if the rpi_armv7l binary was found then that one is used
BENCHMARK_MODEL:="${TF_DIR}/tensorflow/lite/tools/make/gen/${HOST_OS}_${HOST_ARCH}/bin/benchmark_model"
if [ ! -f "${BENCHMARK_MODEL}" ]; then
    echo "${HOST_OS}_${HOST_ARCH} benchmark binary not found. Please build the larq compute engine (tf lite part) first."
    exit 1
fi

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
OUTPUTFILE="benchmarking_results_${current_time}.txt"

if [ -f "$OUTPUTFILE" ]; then
    rm ${OUTPUTFILE}
fi

for MODELFILE in ${ALLMODELS}
do
    echo -e "\nBenchmark for: ${MODELFILE}\n" | tee --append ${OUTPUTFILE}
    ${BENCHMARK_MODEL} --graph="${MODELFILE}" --enable_op_profiling=true | tee --append ${OUTPUTFILE}
done

