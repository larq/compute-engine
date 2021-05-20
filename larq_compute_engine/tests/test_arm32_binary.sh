#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/armhf_linux_toolchain/arm-linux-gnueabihf/libc"

qemu-arm "$1"
