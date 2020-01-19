# Larq Compute Engine

### Setup Docker container

We will build the compute engine inside a Docker container. The image that you use depends on the version of Tensorflow:

- To build the compute engine for Tensorflow 2.x (manylinux2010 compatible) use `custom-op-ubuntu16`
- To build the compute engine for Tensorflow 1.x (not manylinux2010 compatible) use `custom-op-ubuntu14`

You can download and start a docker container as follows:
``` bash
docker pull tensorflow/tensorflow:custom-op-ubuntu16
docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

Inside the Docker container, clone this repository.
``` bash
git clone https://github.com/plumerai/compute-engine.git
cd compute_engine
```

### Build PIP package
You can build the pip package with Bazel.

For the configure script, answer Yes to the manylinux2010 question when you want to build for Tensorflow 2.x or Tensorflow 1.15, and No for Tensorflow 1.14 and below.
``` bash
./configure.sh
bazel build :build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

The wheel file is now available in `artifacts/`, you can install it with
``` bash
pip install artifacts/*.whl
```

### Test the PIP package

You can test the package by running the following python code
```python
import larq_compute_engine as lqce

print(lqce.bsign([[1,-2], [-3,-4]], [[-1,-2], [3,4]]))
```


### Running tests

Run CC unit tests
``` bash
bazel test larq_compute_engine/cc/tests:cc_tests_general
```

Run Python unit tests
``` bash
bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
```

### Benchmarking

See [benchmarking TF lite ops](larq_compute_engine/tflite/benchmark) for benchmarks of complete ops in TF lite.

Benchmarks of sub-components can be run using
``` bash
bazel run larq_compute_engine:benchmark
```

To cross-compile the benchmark program for the Raspberry Pi (armv7), use
``` bash
bazel build larq_compute_engine:benchmark --config=rpi3
```

The resulting binary `bazel-bin/larq_compute_engine/benchmark` can then be copied to the Raspberry Pi to be used.


## TF lite

The core of the TF lite library is a C++ library. There are python, Android and iOS wrappers around it. Note that it is possible to use the C++ library directly on Android as well.

This is independent of the normal tensorflow part of the compute engine. It does not require the docker image or the tensorflow python package.

Please see the [TF lite readme](larq_compute_engine/tflite/build/README.md) for more information.

To benchmark the TF lite ops, see [benchmarking](larq_compute_engine/tflite/benchmark/README.md).

To run unit tests for TF lite, see [TF lite unittests](larq_compute_engine/tflite/python/README.md).
