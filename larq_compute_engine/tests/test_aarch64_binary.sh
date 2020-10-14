#!/bin/bash

set -e

export QEMU_LD_PREFIX="external/aarch64_compiler/aarch64-none-linux-gnu/libc"

qemu-aarch64 "$1"
