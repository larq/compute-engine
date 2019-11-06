#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/arm_compiler/arm-rpi-linux-gnueabihf/sysroot"
qemu-arm larq_compute_engine/bgemm_tests
qemu-arm larq_compute_engine/packbits_tests
qemu-arm larq_compute_engine/fused_bgemm_tests
qemu-arm larq_compute_engine/im2col_tests
qemu-arm larq_compute_engine/bconv2d_tests
