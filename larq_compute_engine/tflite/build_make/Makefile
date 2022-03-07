#
# This is based on
# tensorflow/tensorflow/lite/tools/make/Makefile
#
# The makefile will always be run from the root of the compute engine repository

# Make uses /bin/sh by default, which is incompatible with the bashisms seen
# below.
SHELL := /bin/bash

TF_DIR := third_party/tensorflow
TF_MAKEFILE_DIR := $(TF_DIR)/tensorflow/lite/tools/make

ifeq ($(LCE_GEN_DIR),)
$(error Please set LCE_GEN_DIR to specify an output dir)
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

#LCE: Removed the following includes. It is unclear what they were for.
#-I$(TF_MAKEFILE_DIR)/../../../../../ \
#-I$(TF_MAKEFILE_DIR)/../../../../../../ \
#-I$(OBJDIR)

INCLUDES := \
-Ilarq_compute_engine/tflite/cc \
-I. \
-I$(TF_DIR) \
-I$(TF_MAKEFILE_DIR)/downloads/ \
-I$(TF_MAKEFILE_DIR)/downloads/eigen \
-I$(TF_MAKEFILE_DIR)/downloads/absl \
-I$(TF_MAKEFILE_DIR)/downloads/gemmlowp \
-I$(TF_MAKEFILE_DIR)/downloads/ruy \
-I$(TF_MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(TF_MAKEFILE_DIR)/downloads/farmhash/src \
-I$(TF_MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(TF_MAKEFILE_DIR)/downloads/fp16/include \
-Ilarq_compute_engine/tflite/benchmark/
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz \
-ldl

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CFLAGS := -O3 -DNDEBUG -fPIC $(EXTRA_CFLAGS)
CXXFLAGS := $(CFLAGS) --std=c++14 $(EXTRA_CXXFLAGS)
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

# Added by LCE:
CXXFLAGS += -DTFLITE_WITH_RUY
BUILD_WITH_RUY_PROFILER ?= false
ifeq ($(BUILD_WITH_RUY_PROFILER),true)
  CXXFLAGS += -DRUY_PROFILER
endif

ifeq ($(HOST_OS),windows)
CXXFLAGS += -fext-numeric-literals -D__LITTLE_ENDIAN__
endif

# Auto-detect optimization opportunity if building natively.
ifeq ($(HOST_OS),$(TARGET))
ifeq ($(HOST_ARCH),$(TARGET_ARCH))
ifeq ($(TARGET_ARCH),armv7l)
ifneq ($(shell cat /proc/cpuinfo | grep Features | grep neon),)
  ifneq ($(shell cat /proc/cpuinfo | grep Features | grep vfpv4),)
    CXXFLAGS += -mfpu=neon-vfpv4
  else
    CXXFLAGS += -mfpu=neon
  endif
endif # ifeq ($(TARGET_ARCH),armv7l)
endif # ifeq ($(HOST_ARCH),$(TARGET_ARCH))
endif # ifeq ($(HOST_OS),$(TARGET))
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
CORE_LIB_NAME := libtensorflow-lite.a
BENCHMARK_LIB_NAME := benchmark-lib.a

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

LCE_CORE_SRCS := $(wildcard larq_compute_engine/tflite/kernels/*.cc)

LCE_EXAMPLE_SRCS := \
	examples/lce_minimal.cc

LCE_BENCHMARK_SRCS := \
	larq_compute_engine/tflite/benchmark/lce_benchmark_main.cc

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targeted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(TF_MAKEFILE_DIR)/targets/*_makefile.inc)

# Where compiled objects are stored.
TARGET_OUT_DIR ?= $(TARGET)_$(TARGET_ARCH)
GENDIR := $(TF_MAKEFILE_DIR)/gen/$(TARGET_OUT_DIR)/
OBJDIR := $(GENDIR)obj/
LIBDIR := $(GENDIR)lib/
BINDIR := $(LCE_GEN_DIR)/$(TARGET_OUT_DIR)/

CORE_LIB := $(LIBDIR)$(CORE_LIB_NAME)
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
LCE_EXAMPLE_BINARY := $(BINDIR)lce_minimal
LCE_BENCHMARK_BINARY := $(BINDIR)lce_benchmark

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

LCE_CORE_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(LCE_CORE_SRCS)))))

LCE_EXAMPLE_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(LCE_EXAMPLE_SRCS))))

LCE_BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(LCE_BENCHMARK_SRCS))))

# The target that's compiled if there's no command-line arguments.
all: $(LCE_EXAMPLE_BINARY) $(LCE_BENCHMARK_BINARY)

# For normal manually-created TensorFlow Lite C++ source files.
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow Lite C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LCE_EXAMPLE_BINARY): $(LCE_CORE_OBJS) $(LCE_EXAMPLE_OBJS) $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(LCE_EXAMPLE_BINARY) $(LCE_CORE_OBJS) $(LCE_EXAMPLE_OBJS) \
	$(LIBFLAGS) $(CORE_LIB) $(LDFLAGS) $(LIBS)

$(LCE_BENCHMARK_BINARY): $(LCE_CORE_OBJS) $(LCE_BENCHMARK_OBJS) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(LCE_BENCHMARK_BINARY) $(LCE_CORE_OBJS) $(LCE_BENCHMARK_OBJS) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

# Gets rid of all generated files.
clean:
	rm -rf $(TF_MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
