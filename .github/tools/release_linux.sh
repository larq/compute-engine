#!/usr/bin/env bash
set -e -x

python configure.py

# Build
bazel build :build_pip_pkg \
  --copt=-fvisibility=hidden \
  --copt=-mavx \
  --distinct_host_configuration=false \
  --verbose_failures \
  --crosstool_top=//third_party/toolchains/gcc7_manylinux2010-nvcc-cuda11:toolchain

# Package Whl
bazel-bin/build_pip_pkg artifacts

# Remove manylinux2010 config flags so that normal builds work as expected
rm -f .lce_configure.bazelrc
