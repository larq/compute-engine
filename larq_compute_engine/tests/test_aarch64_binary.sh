#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/aarch64_linux_toolchain/aarch64-linux-gnu/libc"

qemu-aarch64 "$1"
