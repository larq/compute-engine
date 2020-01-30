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

#LQCE: Removed the following includes. It is unclear what they were for.
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
-I$(TF_MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(TF_MAKEFILE_DIR)/downloads/farmhash/src \
-I$(TF_MAKEFILE_DIR)/downloads/flatbuffers/include
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG -fPIC
CXXFLAGS += $(EXTRA_CXXFLAGS)
CFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=c++11
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

ifeq ($(HOST_OS),windows)
CXXFLAGS += -fext-numeric-literals -D__LITTLE_ENDIAN__
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a
BENCHMARK_LIB_NAME := benchmark-lib.a

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

LQCE_ALL_SRCS := \
$(wildcard larq_compute_engine/tflite/cc/*.cc) \
$(wildcard larq_compute_engine/tflite/cc/kernels/*.cc) \

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(TF_MAKEFILE_DIR)/targets/*_makefile.inc)

# Where compiled objects are stored.
GENDIR := $(TF_MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
LIBDIR := $(GENDIR)lib/

LIB_PATH := $(LIBDIR)$(LIB_NAME)
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(LQCE_ALL_SRCS)))))

# For normal manually-created TensorFlow Lite C++ source files.
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

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(BENCHMARK_LIB)

# Add together all the objects we've compiled to the '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)
	# Move the original register.o to the end so that
	# our version, register_lqce.o, has precedence
	$(AR) -m $(LIB_PATH) register.o

# Make sure benchmark_lib depends on lib_path so that the access-times are correct.
# If not, the tflite Makefile will rebuild it
$(BENCHMARK_LIB): $(LIB_PATH) $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS)
	# Move the original register.o to the end
	$(AR) -m $(BENCHMARK_LIB) register.o

lib: $(LIB_PATH)

libdir:
	@echo $(LIBDIR)

# Gets rid of all generated files.
clean:
	rm -rf $(TF_MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
