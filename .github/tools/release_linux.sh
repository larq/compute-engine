#!/usr/bin/env bash
set -e -x

python configure.py

# Build
bazel build :build_pip_pkg \
  --copt=-fvisibility=hidden \
  --copt=-mavx \
  --distinct_host_configuration=false \
  --verbose_failures \
  --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain

# Package Whl
bazel-bin/build_pip_pkg artifacts

# Remove manylinux2014 config flags so that normal builds work as expected
rm -f .lce_configure.bazelrc
