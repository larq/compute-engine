#!/bin/bash

# Exit the script when the test executable fails
set -e

if ! compgen -G "testing_models/*.tflite*" > /dev/null; then
    echo "No models found in testing_models/"
    exit 1
fi
MODELS_TO_TEST=$(ls testing_models/*.tflite)


# We should move all this logic to some common file because it
# is used by `build_lqce.sh` and by `benchmark_model.sh` as well.

HOST_OS="unknown"
UNAME="$(uname -s)"
if [ "$UNAME" = "Linux" ] ; then
    HOST_OS="linux"
elif [ "$UNAME" = "Darwin" ] ; then
    HOST_OS="osx"
fi
HOST_ARCH="$(if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)"

if [ "$HOST_OS" = "linux" ] && [ "$HOST_ARCH" == "armv7l" ]; then
    HOST_OS="rpi"
fi
if [ "$HOST_OS" = "linux" ] && [ "$HOST_ARCH" == "aarch64" ]; then
    HOST_OS="aarch64"
    HOST_ARCH="armv8-a"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
TF_DIR="${ROOT_DIR}/ext/tensorflow"
TEST_EXE="${TF_DIR}/tensorflow/lite/tools/make/gen/${HOST_OS}_${HOST_ARCH}/bin/test_model"

# Check if it exists
if [ ! -f "${TEST_EXE}" ]; then
    echo "Could not find the testing executable: ${TEST_EXE}."
    exit 1
fi

echo "Running model tests."
for MODEL_FILE in ${MODELS_TO_TEST}
do
    ${TEST_EXE} "${MODEL_FILE}"
    if [ $? -ne 0 ]; then
        echo "FAILED: ${MODEL_FILE}"
    fi
done
echo "Model tests completed."
