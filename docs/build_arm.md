# Building Larq Compute Engine for ARM-based boards
This page descibes how to build a Larq Compute Engine (LCE) inference binary
for 32-bit, as well as 64-bit ARM-based boards.
[Bazel](https://bazel.build/) is the primary build
system for LCE. However, since in some occasions Make build system is a more
convenient build solution,  we also provide scripts to build LCE with Make as well.

In this guide, we use the [LCE benchmark tool](../larq_compute_engine/tflite/benchmark)
source to build an inference binary.
See [here](./inference.md) to find out how you can create your own
custom LCE inference programs.

## Cross-compiling LCE with Bazel ##
To cross-compile a LCE inference program for ARM64 architecture,
the bazel target needs to be build with ```--config=aarch64``` flag.
For example, to build the LCE benchmark tool,
run the following command from LCE root directory:

```bash
bazel build -c opt \
    --config=aarch64 \
    //larq_compute_engine/tflite/benchmark:lce_benchmark_model
 ```
To build for ARM32 architecture, use the ```--config=rpi``` flag instead.

## Building LCE with Make ##
To build LCE with Make, first make sure the tensorflow submodule is loaded
(this only has to be done once):
``` bash
git submodule update --init
```
To simplify the build process for various supported targets, we provide
`build_lce.sh` script which accepts the build target platform as an input
argument. The resulting compiled files will be stored in
`third_party/tensorflow/tensorflow/lite/tools/make/gen/<TARGET>/` where,
depending on your target platform, `TARGET` can be `linux_x86_64`, `rpi_armv7l`
or `aarch64_armv8-a`. In the `lib` folder, you can find the Tensorflow Lite
static library `libtensorflow-lite.a` including LCE customs ops.

### Cross-compiling for Raspberry Pi or other ARM based systems with Make ###
To cross-compile LCE, you need to first install the compiler toolchain.
For Debian based systems, run the following commands:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To build for ARM32 architecture, run the following command from
the LCE root directory:
```bash
larq_compute_engine/tflite/build_make/build_lce.sh --rpi
```
When building for ARM64 architecture, replace `--rpi` with `--aarch64`.

### Compile natively with Make ###
To natively build the LCE library and C++ example programs,
first you need to install the compiler toolchain on your target device.
For example, on a Raspberry Pi board, run the following command:
```
sudo apt-get install build-essential
```

You should then be able to natively compile LCE:
```bash
larq_compute_engine/tflite/build_make/build_lce.sh --native
```
NOTE: when building natively on a 32-bit Raspberry Pi, replace `--native` with
`--rpi`.
