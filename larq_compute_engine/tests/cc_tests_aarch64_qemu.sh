#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/aarch64_compiler/aarch64-none-linux-gnu/libc"

for testfile in larq_compute_engine/core/bitpacking/tests/*_test;
do
    qemu-aarch64 "$testfile"
done

for testfile in larq_compute_engine/tflite/tests/*_test;
do
    qemu-aarch64 "$testfile"
done
