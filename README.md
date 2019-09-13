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

### Run CC Unit Tests
``` bash
    bazel test larq_compute_engine:cc_tests
```

### Run Python Tests
``` bash
    bazel test larq_compute_engine:py_tests --python_top=//larq_compute_engine:pyruntime
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
    python -c "import tensorflow as tf;import larq_compute_engine as lce;print(lce.bgemm([[1,2], [3,4]], [[1,2], [3,4]]).eval(session=tf.Session()));"
```
