# Larq Compute Engine

### Setup Docker Container
You are going to build the op inside a Docker container. Pull the provided Docker image from TensorFlow's Docker hub and start a container.

Use the following command if the TensorFlow pip package you are building
against is not yet manylinux2010 compatible:
``` bash
docker pull tensorflow/tensorflow:custom-op-ubuntu14
docker run -it tensorflow/tensorflow:custom-op-ubuntu14 /bin/bash
```
And the following instead if it is manylinux2010 compatible:

``` bash
docker pull tensorflow/tensorflow:custom-op-ubuntu16
docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

Inside the Docker container, clone this repository.
``` bash
git clone https://github.com/plumerai/compute-engine.git
cd compute_engine
```

### Build PIP Package
You can build the pip package with Bazel.

``` bash
./configure.sh
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

### Install and Test PIP Package
Once the pip package has been built, you can install it with,
``` bash
pip install artifacts/*.whl
```

Then test out the pip package
``` bash
cd ..
python -c "import tensorflow as tf;import larq_compute_engine as lqce;print(lce.bgemm([[1,2], [3,4]], [[1,2], [3,4]]).eval(session=tf.Session()));"
```

## Running Tests

### Run CC Unit Tests
``` bash
bazel test larq_compute_engine:cc_tests
```

### Run Python Unit Tests
``` bash
bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
```

### Build TF Lite library

The core of the TF Lite library is a C++ library. There are python, Android and iOS wrappers around this that we will ignore for now. Note that it is possible to use the C++ library directly on Android as well.

This is independent of the normal tensorflow part of the compute engine. It does not require the docker image or the tensorflow python package.

First, make sure the tensorflow submodule is loaded. This only has to be done once.

``` bash
git submodule update --init
```

Now download the tflite dependencies. This only has to be done once.
``` bash
tflite/download_dependencies.sh
```

You can now compile tflite
``` bash
tflite/build_lib.sh
```

To build for a Raspberry Pi, you can either run the above command on a Raspberry Pi, or you can cross-compile from another machine.
To cross-compile make sure the cross compiler is installed:
``` bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
On non-Debian based systems, the package could be called `arm-linux-gnueabihf`.

To cross-compile, simply run
``` bash
tflite/build_rpi_lib.sh
```

The `tflite` directory contains build scripts for other processor architectures as well.
