#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/aarch64_compiler/aarch64-linux-gnu/libc"

for testfile in larq_compute_engine/core/tests/*_tests;
do
    qemu-aarch64 "$testfile"
done
