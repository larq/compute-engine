#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/arm_compiler/arm-rpi-linux-gnueabihf/sysroot"

for testfile in larq_compute_engine/core/tests/*_tests;
do
    qemu-arm "$testfile"
done

for testfile in larq_compute_engine/tflite/tests/*_test;
do
    qemu-arm "$testfile"
done
