
# Building TF lite

First, make sure the tensorflow submodule is loaded. This only has to be done once.

``` bash
git submodule update --init
```

This build system is designed to be independent from the original TF lite build system. We will use the original build scripts whenever possible.

The build tools for TF lite are available in `ext/tensorflow/tensorflow/lite/tools/make`.

Start by downloading the tflite dependencies. This only has to be done once.
``` bash
ext/tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
```

Now we can build the TF lite library

```bash
ext/tensorflow/tensorflow/lite/tools/make/build_lib.sh
```

The resulting compiled files will be stored in `ext/tensorflow/tensorflow/lite/tools/make/gen/TARGET/` where `TARGET` can be `linux_x86_64` or `rpi_armv7l` and so on.

The static library that is produced by this process is `libtensorflow-lite.a`, an archive of compiled `*.o` files.

We will separately compile our own tflite-related files and simply append them to this archive, possibly overwriting certain source files if we would want to.

```bash
larq_compute_engine/tflite/build/build_lqce.sh
```

This will update the file `ext/tensorflow/tensorflow/lite/tools/make/gen/TARGET/lib/libtensorflow-lite.a` with our modifications, which can be used to build the Python, Android or iOS wrapper.


## Building the Python package

The python wrapper package can be built with

``` bash
ext/tensorflow/tensorflow/lite/tools/pip_package/build_pip_package.sh
```

The resulting package file will be saved in `/tmp/tflite_pip/python3/dist/`.


## Building for Android

Please see the [official instructions](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally).


## Building for iOS

Please see the [official instructions](https://www.tensorflow.org/lite/guide/build_ios).

After running `ext/tensorflow/tensorflow/lite/tools/make/build_ios_universal_lib.sh` as instructed, run
```bash
larq_compute_engine/tflite/build/build_lqce.sh
```
Now it will include our modifications.


## Cross-compiling for Raspberry Pi or other ARM based systems

You can either follow the above procedure on a Raspberry Pi, or you can cross-compile from another machine.
To cross-compile make sure the cross compiler is installed:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To cross-compile, run
```bash
ext/tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh
larq_compute_engine/tflite/build/build_lqce.sh
```

The Python package depends on `swig` and currently does not seem to support cross-compiling.
It can be built as follows:

- Compile `libtensorflow-lite.a` as above using `build_rpi_lib.sh` from another machine.
- Copy the cross-compiled `gen/rpi_armv7l/lib/libtensorflow-lite.a` file to a Raspberry Pi and put it in `ext/tensorflow/tensorflow/lite/tools/make/gen/linux_armv7l/lib/`.
- On the Raspberry Pi navigate to `ext/tensorflow/tensorflow/lite/tools/pip_package/`.
- Open `setup.py` and find the class `CustomBuildExt`. Uncomment the line `make()` in the member function `run`.
- Now run `./build_pip_package.sh`.

