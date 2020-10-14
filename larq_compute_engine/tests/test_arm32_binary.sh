#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/arm_compiler/arm-rpi-linux-gnueabihf/sysroot"

qemu-arm "$1"
