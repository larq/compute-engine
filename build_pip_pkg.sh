#!/usr/bin/env bash
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
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_linux() {
    [[ "${PLATFORM}" == "linux" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

if is_windows; then
  # On windows, the workspace name is lce to avoid the path length limit of MSVC
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.exe.runfiles/lce/"
else
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/larq_compute_engine/"
fi

function abspath() {
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  cd "$OLDPWD"
}

function main() {
  DEST=${1}
  BUILD_FLAG=${@:2}

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  mkdir -p ${DEST}
  DEST=$(abspath "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy Larq Compute Engine files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}README.md "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  if is_linux; then
    touch ${TMPDIR}/stub.cc
  fi
  if is_windows; then
    from=$(cygpath -w ${PIP_FILE_PREFIX}larq_compute_engine)
    to=$(cygpath -w "${TMPDIR}"/larq_compute_engine)
    start robocopy //S "${from}" "${to}" //xf *_test.py
    sleep 5
  else
    rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}larq_compute_engine "${TMPDIR}"
  fi
  pushd ${TMPDIR}

  if ! is_windows; then
    echo "=== Stripping symbols"
    chmod +w ${TMPDIR}/larq_compute_engine/mlir/*.so ${TMPDIR}/larq_compute_engine/tflite/python/*.so
    strip -x ${TMPDIR}/larq_compute_engine/mlir/*.so ${TMPDIR}/larq_compute_engine/tflite/python/*.so
  fi

  echo $(date) : "=== Building wheel"
  python setup.py bdist_wheel ${BUILD_FLAG} > /dev/null

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
