#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2020 Larq Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e
shopt -s expand_aliases
alias pip='pip3'
alias python='python3'
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function is_linux() {
    [[ "${PLATFORM}" == "linux" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if we are building against manylinux1 or manylinux2010 pip package,
# default manylinux2010
if ! is_linux; then
  echo "On Windows or Mac, skipping toolchain flags.."
  PIP_MANYLINUX2010=0
else
  while [[ "$PIP_MANYLINUX2010" == "" ]]; do
    read -p "Does the pip package have tag manylinux2010 (usually the case for nightly release after Aug 1, 2019, or official releases past 1.14.0)?. Y or enter for manylinux2010, N for manylinux1. [Y/n] " INPUT
    case $INPUT in
      [Yy]* ) PIP_MANYLINUX2010=1;;
      [Nn]* ) PIP_MANYLINUX2010=0;;
      "" ) PIP_MANYLINUX2010=1;;
      * ) echo "Invalid selection: " $INPUT;;
    esac
  done
fi

if is_linux; then
  write_to_bazelrc "build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"
fi
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

if [[ "$PIP_MANYLINUX2010" == "1" ]]; then
  if is_linux; then
    write_to_bazelrc "build --config=manylinux2010"
    write_to_bazelrc "test --config=manylinux2010"
  fi
fi

if ! is_windows; then
  # By default, build TF in C++ 14 mode.
  write_to_bazelrc "build --cxxopt=-std=c++14"
  write_to_bazelrc "build --host_cxxopt=-std=c++14"
fi


cat << EOM >> .bazelrc
# Disable visibility checks (works around some private deps in TensorFlow).
build --nocheck_visibility

build --copt=-DTFLITE_WITH_RUY

# These can be activated using --config=rpi3 and --config=aarch64

build:rpi3 --crosstool_top=@local_config_arm_compiler//:toolchain
build:rpi3 --cpu=armeabi
build:rpi3 -c opt  \
  --copt=-march=armv7-a --copt=-mfpu=neon-vfpv4 \
  --copt=-std=gnu++11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR \
  --copt=-O3 --copt=-fno-tree-pre \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 \
  --define=raspberry_pi_with_neon=true \
  --define=framework_shared_object=false \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer \
  --verbose_failures

build:aarch64 --crosstool_top=@local_config_arm_compiler//:toolchain
build:aarch64 --cpu=aarch64
build:aarch64 -c opt  \
  --copt=-march=armv8-a \
  --copt=-std=gnu++11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR \
  --copt=-O3 --copt=-fno-tree-pre \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 \
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 \
  --define=framework_shared_object=false \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer \
  --verbose_failures

# Options to build TensorFlow 1.x or 2.x.
build:v1 --define=tf_api_version=1
build:v2 --define=tf_api_version=2
test:v1 --action_env=TF2_BEHAVIOR=0
test:v2 --action_env=TF2_BEHAVIOR=1

# Options to disable default on features
build:noaws --define=no_aws_support=true
build:nogcp --define=no_gcp_support=true
build:nohdfs --define=no_hdfs_support=true
build:nonccl --define=no_nccl_support=true

build --config=v2 --config=noaws --config=nogcp --config=nohdfs --config=nonccl
test --config=v2

# Android configs. Bazel needs to have --cpu and --fat_apk_cpu both set to the
# target CPU to build transient dependencies correctly. See
# https://docs.bazel.build/versions/master/user-manual.html#flag--fat_apk_cpu
build:android --crosstool_top=//external:android/crosstool
build:android --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
build:android_arm --config=android
build:android_arm --cpu=armeabi-v7a
build:android_arm --fat_apk_cpu=armeabi-v7a
build:android_arm64 --config=android
build:android_arm64 --cpu=arm64-v8a
build:android_arm64 --fat_apk_cpu=arm64-v8a
build:android_x86 --config=android
build:android_x86 --cpu=x86
build:android_x86 --fat_apk_cpu=x86
build:android_x86_64 --config=android
build:android_x86_64 --cpu=x86_64
build:android_x86_64 --fat_apk_cpu=x86_64

# The default android SDK/NDK paths are hardcoded here and the user needs
# to change the paths according to the local configuration or
# the "install_android.sh" script
build --action_env ANDROID_NDK_HOME="/tmp/lce_android/ndk/19.2.5345600/"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="27.0.3"
build --action_env ANDROID_SDK_API_LEVEL="28"
build --action_env ANDROID_SDK_HOME="/tmp/lce_android"

EOM
