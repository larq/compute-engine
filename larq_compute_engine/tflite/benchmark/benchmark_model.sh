#!/bin/bash
set -e

usage()
{
    echo "Usage: benchmark_model.sh [--model <model-file>] [--exec <benchmarking-executable>]

--model        Neural network to benchmark. If no benchmarking model is passed,
               the models can either be generated or the pre-generated model will 
               be used.

--exec         Exectuable which performs the benchmarking.
"
}

BENCH_MODEL=""
BENCH_EXE=""

while [ "$1" != "" ]; do
    case $1 in
        -m | --model)
            shift
            BENCH_MODEL=$1
            ;;
        -e | --exec)
            shift
            BENCH_EXE=$1
            ;;
        -h | --help )
            usage
            exit
            ;;
    esac
    shift
done

######################################################################
# SET/GENERATE/FIND THE MODELS TO BENCHMARK
######################################################################
if [ "$BENCH_MODEL" != "" ]; then
    MODELS_TO_BENCHMARK=$BENCH_MODEL
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
    MODELS_TO_BENCHMARK=$(ls benchmarking_models/*.tflite)
fi

######################################################################
# TRY TO SET THE BENCHMARKNIG EXECUTABLE AUTOMATICALLY IF NOT SET BY USER
######################################################################
if [ "$BENCH_EXE" = "" ]; then
    echo -e "\n-> Detecting the benchmarking executable "
    # Try to figure out the host system
    HOST_OS="unknown"
    UNAME="$(uname -s)"
    if [ "$UNAME" = "Linux" ] ; then
        HOST_OS="linux"
    elif [ "$UNAME" = "Darwin" ] ; then
        HOST_OS="osx"
    fi
    HOST_ARCH="$(if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)"

    if [ "$HOST_OS" = "linux" ] && [ "$HOST_ARCH" == "armv7l" ]; then
        # Probably this is a 32-bit Raspberry Pi
        # In this case check the rpi_armv7l folder instead
        HOST_OS="rpi"
    fi
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="${SCRIPT_DIR}/../../.."
    TF_DIR="${ROOT_DIR}/ext/tensorflow"
    BENCH_EXE="${TF_DIR}/tensorflow/lite/tools/make/gen/${HOST_OS}_${HOST_ARCH}/bin/benchmark_model"
    echo -e "\n-> $BENCH_EXE"

    # CHECK THE EXECUTABLE
    if [ ! -f "${BENCH_EXE}" ]; then
        echo -e "\n-> Could not find the benchmarking executable: ${BENCH_EXE}."
        exit 1
    fi
else
    echo -e "\n-> Using benchmarking executable: $BENCH_EXE"
fi

######################################################################
# SETTING OUTPUT FILES
######################################################################
current_time=$(date "+%Y_%m_%d-%H_%M_%S")
OUTPUT_FILE="benchmarking_results_${current_time}.txt"
OUTPUT_FILE_SUMMARY="benchmarking_results_${current_time}_summary.txt"

if [ -f "$OUTPUT_FILE" ]; then
    rm ${OUTPUT_FILE}
fi

######################################################################
# PERFORM BENCHMARKING FOR ALL MODELS
######################################################################
OUTPUT_PREFIX="-> Using model"
for MODEL_FILE in ${MODELS_TO_BENCHMARK}
do
    echo -e "\n${OUTPUT_PREFIX}: ${MODEL_FILE}\n" | tee --append ${OUTPUT_FILE} ${OUTPUT_FILE_SUMMARY}
    ${BENCH_EXE} --graph="${MODEL_FILE}" --enable_op_profiling=true | tee --append ${OUTPUT_FILE} | grep -e LqceBconv -e CONV | tee --append ${OUTPUT_FILE_SUMMARY}
done
