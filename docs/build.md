# Building Larq Compute Engine #

The Larq Compute Engine (LCE) repository consists of two main components:
LCE for [TensorFlow](#LCE-for-TensorFlow) and LCE for [TensorFlow Lite](#LCE-for-TensorFlow-Lite).
Each of these components are a collection of optimized ops for TensorFlow and
TensorFlow Lite.
These two components can be built indepent of each other. Below we describe
the build process for each of these components.

### Setup Docker container ###
We will build the LCE inside a [Docker](https://www.docker.com/) container.
To be able to build LCE and the LCE converter's
[`manylinux2010`](https://www.python.org/dev/peps/pep-0571/) compatible PIP
package, we use the [`tensorflow/tensorflow:custom-op-ubuntu16`](https://hub.docker.com/r/tensorflow/tensorflow) image. 
For macOS, make sure to have a C++ compiler installed.

You can clone the LCE repository in a directoy of the
host machine and mount that directroy as a
[volume]((https://docs.docker.com/storage/volumes/)) in the LCE container:
``` bash
# create the docker shared volume directory
mkdir lce-volume
git clone https://github.com/larq/compute-engine.git lce-volume

# download and start a docker container
docker pull tensorflow/tensorflow:custom-op-ubuntu16
docker run -it -v $PWD/lce-volume:/tmp/lce-volume -w /tmp/lce-volume tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```
the `lce-volume` in the host machine is now mapped to the `/tmp/lce-volume/`directory
inside the container.

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
"Yes" to the ```manylinux2010``` question if you want to build the
LCE converter's PIP package for TensorFlow 2.x.

## LCE for TensorFlow Lite ##
LCE for TensorFlow Lite has a diverse platform support, covering
[Android](./quickstart_android.md), [ARM-based boards](./build_arm.md)
such as Raspberry Pi and x86 machines. To build/install/run LCE on
each of these platforms, please refer to the corresponding guide.

## LCE for TensorFlow ##

LCE provides a Python PIP package for its TensorFlow Ops.
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

To run all python unittests of LCE for TensorFlow, execute the following
bazel command:
``` bash
bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
```
