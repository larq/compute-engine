#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/arm_compiler/arm-rpi-linux-gnueabihf/sysroot"

for testfile in larq_compute_engine/*_tests;
do
    qemu-arm "$testfile"
done
