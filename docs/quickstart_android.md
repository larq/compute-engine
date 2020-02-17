# Larq Compute Engine Android Quickstart

To build Larq Compute Engine (LCE) for Android,
you must have the [Android NDK](https://developer.android.com/ndk) and
[SDK](https://developer.android.com/studio) installed on your system.
Below we explain how to install the Android prerequisites in the LCE
Docker container and how to configure the LCE Bazel build settings
accordingly. Before proceeding with the next steps, please follow
the instructions in the main [LCE build guide](/compute-engine/build) to setup the
Docker container for LCE and the Bazel build system.

NOTE: we recommend using the docker volume as described in the
[LCE build guide](/compute-engine/build) to be able to easily transfer
files in-between the container and host machine.

#### Install prerequisites

We provide a bash script which uses the `sdkmanager` tool
to install the Android NDK and SDK inside the Docker container.
Please run the script by executing the following command from the LCE
root directory:

```bash
./third_party/install_android.sh
```

After executing the bash script, please accept the Android SDK licence agreement.
The script will download and unpack the android NDK and SKD under the directory
`/tmp/lce_android` in the LCE docker container.

#### Custom android version

The Android NDK and SDK versions used in LCE are currently hard-coded in the
install script.
To build LCE against a different NDK and SDK versions, you can manually
modify `ANDROID_VERSION` and `ANDROID_NDK_VERSION` variables in the
install script. Additionally, the following configurations in `configure.sh`
(or `.bazelrc`) file need to be adjusted:

```shell
build --action_env ANDROID_NDK_HOME="/tmp/lce_android/ndk/19.2.5345600"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

#### Build an LCE inference binary

To build an LCE inference binary for Android (see [here](/compute-engine/inference) for creating your
own LCE binary) the Bazel target needs to build with `--config=android_arm64` flag.

To build the [minimal example](https://github.com/larq/compute-engine/blob/master/examples/lce_minimal.cc) for Android,
run the following command from the LCE root directory:

```bash
bazel build -c opt \
    --config=android_arm64 \
    //examples:lce_minimal
```

To build the [LCE benchmark tool](https://github.com/larq/compute-engine/tree/master/larq_compute_engine/tflite/benchmark)
for Android, simply build the bazel target
`//larq_compute_engine/tflite/benchmark:lce_benchmark_model`.

The resulting binaries will be stored at
`bazel-bin/examples/lce_minimal` and
`bazel-bin/larq_compute_engine/tflite/benchmark/lce_benchmark_model`.
See below on how to copy them to your android device.

#### Run inference

To run the inference with a [Larq converted model](/compute-engine/converter) on an android phone,
please follow these steps on your host machine (replace the `lce_benchmark_model` with your
desired inference binary):

(1) Install the [Android Debug Bridge (adb)](https://developer.android.com/studio/command-line/adb) on your host machine.

(2) Follow the instructions [here](https://developer.android.com/studio/debug/dev-options#enable)
to enable `USB debugging` on your android phone.

(3) Connect your phone and run the following command to confirm that your host
computer is connected to the phone:

```bash
adb devices
```

(4) Since Bazel stores the build artifacts outside the `lce-volume` directory,
You need to run the following command inside the container to copy the inference
binary from the bazel output direcotry to the `lce-volume` directory in order to be able to access it on the host
machine:

```bash
cp bazel-bin/larq_compute_engine/tflite/benchmark/lce_benchmark_model <volume-dir>
```

(5) Transfer the LCE inference binary to your phone:

```bash
adb push lce_benchmark_model  /data/local/tmp
```

(6) Make the binary executable:

```shell
adb shell chmod +x /data/local/tmp/lce_benchmark_model
```

(7) Transfer the converted `.tflite` model file to your phone:

```shell
adb push binarynet.tflite /data/local/tmp
```

(8) Run the inference:

```shell
adb shell /data/local/tmp/lce_benchmark_model \
    --graph=/data/local/tmp/binarynet.tflite \
    --num_threads=4
```
