
# Building TF lite

First, make sure the tensorflow submodule is loaded. This only has to be done once.

``` bash
git submodule update --init
```

This build system is designed to be as independent from the original TF lite build system as possible. We will use the original build scripts whenever possible.

The build tools for TF lite are available in `tensorflow/tensorflow/lite/tools/make`.

Start by downloading the tflite dependencies. This only has to be done once.
``` bash
tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
```

Now we can build the TF lite library

```bash
tensorflow/tensorflow/lite/tools/make/build_lib.sh
```

The resulting compiled files will be stored in `tensorflow/tensorflow/lite/tools/make/gen/TARGET/` where `TARGET` can be `linux_x86_64` or `rpi_armv7l` and so on.
In this directory there is a `Makefile` and scripts like `build_rpi_lib.sh` that run the `Makefile` with the correct parameters for the target architecture.

The library that is produced by this process is `libtensorflow-lite.a`, an archive of compiled `*.o` files.

We will separately compile our own tflite-related files and simply append them to this archive, possibly overwriting certain source files if we would want to.

```bash
larq_compute_engine/tflite/build/build_lqce.sh
```

This will update the file `tensorflow/tensorflow/lite/tools/make/gen/TARGET/lib/libtensorflow-lite.a` with our modifications, which can be used to build C++ examples or the Python wrapper.


## Compiling for Raspberry Pi or other ARM based processors

To build for a Raspberry Pi, you can either run the above `build_lib.sh` command on a Raspberry Pi, or you can cross-compile from another machine.
To cross-compile make sure the cross compiler is installed:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To cross-compile, simply replace `build_lib.sh` by `build_rpi_lib.sh`. That directory contains scripts for other processor architectures as well.


## Building the Python package

The python wrapper package can be built with

``` bash
tensorflow/tensorflow/lite/tools/pip_package/build_pip_package.sh
```

The resulting package will be saved in `/tmp/tflite_pip/`.
This build process depends on `swig` and currently it seems that this does not support crosscompiling. The Python package for Raspberry Pi should therefore be built on an actual Raspberry Pi. The compilation of the `libtensorflow-lite.a` can still be done on another machine.


## Building for Android

Please see the [official instructions](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally) for building TF lite for android.
