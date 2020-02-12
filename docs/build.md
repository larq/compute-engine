# Building Larq Compute Engine #

The Larq Compute Engine (LCE) repository consists of two main components:
LCE for [Tensorflow](#LCE-for-Tensorflow) and LCE for [Tensorflow Lite](#LCE-for-Tensorflow-Lite).
Each of these components are a collection of optimzied ops for Tensorflow and
Tensorflow Lite.
These two components can be built indepent of each other. Below we describe
the build process for each of these components.

### Setup Docker container ###
We will build the LCE inside a Docker container.
The image that you use depends on the version of Tensorflow:

- To build the LCE for Tensorflow 2.x (`manylinux2010` compatible)
  use `custom-op-ubuntu16`
- To build the LCE for Tensorflow 1.x (not manylinux2010 compatible)
  use `custom-op-ubuntu14`

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

Alternatively, you can clone the LCE repository in a directoy of the
host machine and mount that directroy as a volume in the LCE container
as described [here](https://docs.docker.com/storage/volumes/).

### Install Bazel ###

[Bazel](https://bazel.build/) is the primary build system for LCE.
However, to avoid Bazel compatibility issues,
we recommend to use [Bazelisk](https://github.com/bazelbuild/bazelisk).
To install Bazelisk, run the following command on Linux
(replace the ```v1.2.1``` with your desired
[bazelisk version](https://github.com/bazelbuild/bazelisk/releases)):

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
Tensorflow 2.x or Tensorflow 1.15, and "No" for Tensorflow 1.14 and below.

## LCE for Tensorflow Lite ##
LCE for Tensorflow Lite has a diverse platform support, covering
[Android](./quickstart_android.md), [ARM-based boards](./quickstart_arm.md)
such as Raspberry Pi and x86 machines. To build/install/run LCE on
each of these platforms, please refer to the corresponding guide.

## LCE for Tensorflow ##

LCE provides a Python PIP package for its Tensorflow Ops.
You can build the LCE pip package with bazel:
``` bash
bazel build :build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

The wheel file will be stored in `artifacts/` directory, you can install
it with:
``` bash
pip install artifacts/*.whl
```

The installed LCE PIP package can be tested by running the following
python code:
```python
import larq_compute_engine as lce
print(lce.bsign([[1,-2], [-3,-4]], [[-1,-2], [3,4]]))
```

To run the entire python unittests of LCE for Tensorflow, execute the following
bazel command:
``` bash
bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
```
