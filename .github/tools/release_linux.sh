#!/usr/bin/env bash
set -e -x

python configure.py

# Build
# `release_cpu_linux` will activate absolute paths to files that only exist in the tensorflow/build:2.16-pythonXX docker container
bazel build :build_pip_pkg \
  --config=release_cpu_linux \
  --copt=-fvisibility=hidden \
  --copt=-mavx \
  --verbose_failures

# Package Whl
bazel-bin/build_pip_pkg artifacts

# Remove manylinux2014 config flags so that normal builds work as expected
rm -f .lce_configure.bazelrc
