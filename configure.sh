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
  echo "$1" >> .lce_configure.bazelrc
}

function is_linux() {
    [[ "${PLATFORM}" == "linux" ]]
}

# Remove .bazelrc if it already exist
[ -e .lce_configure.bazelrc ] && rm .lce_configure.bazelrc

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
  write_to_bazelrc "build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
fi

if [[ "$PIP_MANYLINUX2010" == "1" ]]; then
  if is_linux; then
    write_to_bazelrc "build --config=manylinux2010"
    write_to_bazelrc "test --config=manylinux2010"
  fi
fi
