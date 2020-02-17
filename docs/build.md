# Build Larq Compute Engine

The Larq Compute Engine (LCE) repository consists of two main components:

- **LCE Runtime:** which is a collection of highly optimized
  [TensorFlow Lite](https://www.tensorflow.org/lite) custom operators.

- **LCE Converter:** which takes a Larq model and generates a TensorFlow Lite
  [FlatBuffer](https://google.github.io/flatbuffers/) file (`.tflite`) compatible
  with LCE runtime.

Before proceeding with building LCE components, you need to setup the LCE
build enviroment first.

## Setup the build environment

### 1. Setup Docker container

We build the Larq Compute Engine (LCE) components inside a
[docker](https://www.docker.com/) container. We also recommend to use
[docker volumes](https://docs.docker.com/storage/volumes/)
to migrate the build targets in-between the host machine and the container.

To be able to build the LCE runtime and the LCE converter's
[`manylinux2010`](https://www.python.org/dev/peps/pep-0571/) compatible PIP
package, we need to use the [`tensorflow:custom-op-ubuntu16`](https://hub.docker.com/r/tensorflow/tensorflow)
image.

First, download the docker image:

```bash
docker pull tensorflow/tensorflow:custom-op-ubuntu16
```

Clone the LCE repository in the host machine:

```bash
mkdir lce-volume
git clone https://github.com/larq/compute-engine.git lce-volume
```

To start the container and map the `lce-volume` directory to the `/tmp/lce-volume`
directory inside the container:

```bash
docker run -it -v $PWD/lce-volume:/tmp/lce-volume \
    -w /tmp/lce-volume tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

Now, you will be able to build your targets inside the container
and access the build artifacts directly from the host machine.

### 2. Install Bazelisk

[Bazel](https://bazel.build/) is the primary build system for LCE.
However, to avoid Bazel compatibility issues,
we recommend to use [Bazelisk](https://github.com/bazelbuild/bazelisk).
To install Bazelisk on Linux, run the following two commands
(replace `v1.2.1` with your preferred
[bazelisk version](https://github.com/bazelbuild/bazelisk/releases)):

```shell
sudo wget -O /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```
### 3. Configure Bazel ###

Run the ```./configure.sh``` script in the LCE root directory and answer
"Yes" to the ```manylinux2010``` question if you want to build the
LCE converter's PIP package inside the `tensorflow:custom-op-ubuntu16`
container. This script generates the Bazel configuration file `.bazelrc`
in the LCE root directory.

## Build LCE Runtime

LCE runtime has a diverse platform support, covering
[Android](https://docs.larq.dev/compute-engine/quickstart_android) and [ARM-based boards](https://docs.larq.dev/compute-engine/build_arm)
such as Raspberry Pi. To build/install/run LCE runtime on
each of these platforms, please refer to the corresponding guide.

## Build LCE Converter

The LCE converter is available on [PyPI](https://pypi.org/project/larq-compute-engine/)
and can be installed with Python's [pip](https://pip.pypa.io/en/stable/)
package manager:

```shell
pip install larq-compute-engine
```

You can also run the following commands,
to build the LCE pip package yourself:

```bash
bazel build :build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

The script stores the wheel file in the `artifacts/` directory located in the
LCE root directory. To install the PIP package:

```bash
pip install artifacts/*.whl
```
