#!/usr/bin/env bash
set -e -x

python configure.py

# Inside the docker container on github actions there is not
# enough space for the bazel cache, but a larger disk is mounted at /github_disk
# so we tell bazel to store everything there

# `release_cpu_linux` will activate absolute paths to files that only exist in the tensorflow/build:2.16-pythonXX docker container
bazel --output_user_root=/github_disk/bazel_root \
  build :build_pip_pkg \
  -c opt \
  --config=release_cpu_linux \
  --copt=-fvisibility=hidden \
  --verbose_failures

# Package Whl
bazel-bin/build_pip_pkg artifacts

# Remove manylinux2014 config flags so that normal builds work as expected
rm -f .lce_configure.bazelrc
