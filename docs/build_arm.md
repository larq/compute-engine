# Building Larq Compute Engine for ARM-based systems
This page descibes how to build a Larq Compute Engine (LCE) inference binary
for 32-bit, as well as 64-bit ARM-based systems. You can either build LCE
natively on your ARM device or cross-compile it from a non-ARM machine.
[Bazel](https://bazel.build/) is the primary build system for LCE, and it can
be used to cross-compile a binary for ARM-based systems using a non-ARM host.
To natively build on an ARM system itself we provide a script that uses the
Makefile build system.

This leaves us with three ways to build LCE, which we recommend in the
following order:
1. To cross-compile LCE from a host machine, see the section
   [Cross-compiling LCE with Bazel](#cross-compiling-lce-with-bazel)
2. To natively compile LCE, see the section
   [Building LCE with Make](#building-lce-with-make).
3. To cross-compile LCE using the Make system for users that do not wish to
   install Bazel, see the section
   [Cross-compiling LCE with Make](#cross-compiling-lce-with-make).

In this guide, we use the [LCE benchmark tool](../larq_compute_engine/tflite/benchmark)
source to build an inference binary. See [here](./inference.md) to find out how
you can create your own custom LCE inference application.

Although the Raspberry Pi 3 and Raspberry Pi 4 have 64-bit CPUs, note that the
popular distribution Raspbian for the Raspberry Pi is a 32-bit OS. In order to
use the optimized 64-bit kernels of LCE on a Raspberry Pi, a 64-bit OS such as
[Manjaro](https://manjaro.org/download/#raspberry-pi-4-xfce) should be used.

## Cross-compiling LCE with Bazel

First configure Bazel using the instructions [here](build.md). Make sure you
have run the `./configure.sh` script as instructed.

To cross-compile an LCE inference binary for ARM architectures, the bazel
target needs to be built with the `--config=rpi3` (32-bit ARM) or
`--config=aarch64` (64-bit ARM) flag. For example, to build the LCE benchmark
tool, run the following command from the LCE root directory:

```bash
bazel build \
    --config=aarch64 \
    //larq_compute_engine/tflite/benchmark:lce_benchmark_model
 ```

The resulting binary will be stored at
`bazel-bin/larq_compute_engine/tflite/benchmark/lce_benchmark_model`. You can
copy this file to your ARM machine and run it there.

## Building LCE with Make
To build LCE with Make, first make sure the tensorflow submodule is loaded
(this only has to be done once):
``` bash
git submodule update --init
```
To simplify the build process for various supported targets, we provide the
`build_lce.sh` script which accepts the build target platform as an input
argument.

To natively build the LCE library and C++ example programs, first you need to
install the compiler toolchain on your target device. For example, on a
Raspberry Pi board with Raspbian, run the following command:
```
sudo apt-get install build-essential
```

You should then be able to natively compile LCE by running the following from
the LCE root directory:
```bash
larq_compute_engine/tflite/build_make/build_lce.sh --native
```

It is also possible to replace `--native` by `--rpi` (32-bit ARM) or
`--aarch64` (64-bit ARM) to add extra compiler optimization flags.

The resulting compiled files will be stored in
`third_party/tensorflow/tensorflow/lite/tools/make/gen/<TARGET>` where,
depending on your target platform, `<TARGET>` can be `linux_x86_64`,
`rpi_armv7l`, or `aarch64_armv8-a`. In the `bin` folder, you can find the
benchmark program `benchmark_model`. In the `lib` folder, you can find the
TensorFlow Lite static library `libtensorflow-lite.a` which includes the LCE
customs ops.

## Cross-compiling LCE with Make
First make sure the tensorflow submodule is loaded (this only has to be done
once):
``` bash
git submodule update --init
```

To cross-compile LCE, you need to first install the compiler toolchain.
For Debian based systems, run the following commands:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To build for 32-bit ARM architectures, run the following command from the LCE
root directory:
```bash
larq_compute_engine/tflite/build_make/build_lce.sh --rpi
```
When building for a 64-bit ARM architecture, replace `--rpi` with `--aarch64`.

See `(Building LCE with Make)(#building-lce-with-make) for the location of
the resulting build files. Copy the `benchmark_model` program to your ARM
machine to run it.
