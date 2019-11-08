
# Building TF lite

First, make sure the tensorflow submodule is loaded. This only has to be done once.

``` bash
git submodule update --init
```

To build the library and C++ example programs, run

```bash
larq_compute_engine/tflite/build/build_lqce.sh --native
```

The resulting compiled files will be stored in `ext/tensorflow/tensorflow/lite/tools/make/gen/TARGET/` where `TARGET` can be `linux_x86_64`, `rpi_armv7l`, `aarch64_armv8-a` and so on.

In the `bin` folder there is the `benchmark_model` example program that can be used to benchmark models and measure times of individual ops.

In the `lib` folder is the static library `libtensorflow-lite.a` which can be used to build the Python package.

When building natively on a 32-bit Raspberry Pi, replace `--native` by `--rpi`.


## Building the Python package

The python wrapper package can be built with

``` bash
larq_compute_engine/tflite/build/build_lqce.sh --native --pip
```

The resulting package file will be saved in `/tmp/tflite_pip/python3/dist/`.

When building natively on a 32-bit Raspberry Pi, replace `--native` by `--rpi`.


## Building for Android

Please see the [official instructions](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally).


## Building for iOS

Please see the [official instructions](https://www.tensorflow.org/lite/guide/build_ios) for how to get the iOS SDK.

Run
```bash
larq_compute_engine/tflite/build/build_lqce.sh --ios
```
to build the iOS library.
The compiled files will be stored in `ext/tensorflow/tensorflow/lite/tools/make/gen/ios_ARCH/` where `ARCH` can be `x86_64`, `armv7`, `armv8s` or `arm64`.


## Cross-compiling for Raspberry Pi or other ARM based systems

To cross-compile make sure the cross compiler is installed:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To cross-compile, run
```bash
larq_compute_engine/tflite/build/build_lqce.sh --rpi
```

The Python package depends on `swig` and currently does not seem to support cross-compiling.
It can be built as follows:

- Compile `libtensorflow-lite.a` as above using `build_rpi_lib.sh --rpi` from another machine.
- Copy the cross-compiled `gen/rpi_armv7l/lib/libtensorflow-lite.a` file to a Raspberry Pi and put it in `ext/tensorflow/tensorflow/lite/tools/make/gen/linux_armv7l/lib/`.
- On the Raspberry Pi navigate to `ext/tensorflow/tensorflow/lite/tools/pip_package/`.
- Open `setup.py` and find the class `CustomBuildExt`. Uncomment the line `make()` in the member function `run`.
- Now run `./build_pip_package.sh`.

