# Building Larq Compute Engine #

The Larq Compute Engine repository consists of two main components: 
LCE for [Tensorflow](#LCE-for-Tensorflow) and [Tensorflow Lite](#LCE-for-Tensorflow-Lite)
which are collection of optimzied ops for Tensorflow and Tensorflow Lite, accordingly. 
These two components can be built indepent of each other. Below we describe the build
process for each of these components.

### Setup Docker container ###
We will build the LCE inside a Docker container. 
The image that you use depends on the version of Tensorflow:

- To build the LCE for Tensorflow 2.x (manylinux2010 compatible) use `custom-op-ubuntu16`
- To build the LCE for Tensorflow 1.x (not manylinux2010 compatible) use `custom-op-ubuntu14`

You can download and start a docker container as follows:
``` bash
docker pull tensorflow/tensorflow:custom-op-ubuntu16
docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

Inside the Docker container, clone this repository:
``` bash
git clone https://github.com/plumerai/compute-engine.git
cd compute_engine
```

Alternatively, you can clone the LCE repository in a directoy of the host machine
and mount that directroy as a volume in the LCE container as described [here](https://docs.docker.com/storage/volumes/).

### Install Bazel ###

LCE uses [bazel](https://bazel.build/) to build and test the software components.
To avoid bazel compatibility issues, we recommend to use [Bazelisk](https://github.com/bazelbuild/bazelisk).
To install Bazelisk, run the following command on Linux (replace the ```v1.2.1``` with the your desired [bazelisk released version](https://github.com/bazelbuild/bazelisk/releases)):

```shell
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

On macOS, run the follwing command:
```
brew install bazelbuild/tap/bazelisk
```

### Configure .bazelrc ###
Run the ```./configure.sh``` script in the root directory and answer 
"Yes" to the ```manylinux2010``` question when you want to build for
Tensorflow 2.x or Tensorflow 1.15, and No for Tensorflow 1.14 and below.

## LCE for Tensorflow Lite ##
<!-- The core of the TF lite library is a C++ library.  -->
<!-- There are python, Android and iOS wrappers around it. -->
<!-- Note that it is possible to use the C++ library directly on Android as well. -->

<!-- This is independent of the normal tensorflow part of the compute engine. It does not require the docker image or the tensorflow python package. -->
<!-- Please see the [TF lite readme](larq_compute_engine/tflite/build/README.md) for more information. -->
<!-- To benchmark the TF lite ops, see [benchmarking](larq_compute_engine/tflite/benchmark/README.md). -->
<!-- To run unit tests for TF lite, see [TF lite unittests](larq_compute_engine/tflite/python/README.md). -->

## LCE for Tensorflow ##

### Build LCE Ops PIP package ###
LCE provides a Python PIP package for its Tensorflow Ops. You can build the LCE pip package with bazel:
``` bash
bazel build :build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

The wheel file will be stored in `artifacts/` directory, you can install it with:
``` bash
pip install artifacts/*.whl
```

The installed LCE PIP package can be tested by running the following python code:
```python
import larq_compute_engine as lce
print(lce.bsign([[1,-2], [-3,-4]], [[-1,-2], [3,4]]))
```

### Running unittests ###
<!-- You can run the C++ unittests of the LCE with the following bazel command: -->
<!-- ``` bash -->
<!-- bazel test larq_compute_engine/core/tests:cc_tests_general -->
<!-- ``` -->

To run the entire python unit tests of LCE, execute the following bazel command:
``` bash
bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
```

<!-- ### Benchmarking ### -->

<!-- See [benchmarking TF lite ops](larq_compute_engine/tflite/benchmark) for benchmarks of complete ops in TF lite. -->

<!-- Benchmarks of sub-components can be run using -->
<!-- ``` bash -->
<!-- bazel run larq_compute_engine:benchmark -->
<!-- ``` -->

<!-- To cross-compile the benchmark program for the Raspberry Pi (armv7), use -->
<!-- ``` bash -->
<!-- bazel build larq_compute_engine:benchmark --config=rpi3 -->
<!-- ``` -->

<!-- The resulting binary `bazel-bin/larq_compute_engine/benchmark` can then be copied to the Raspberry Pi to be used. -->
