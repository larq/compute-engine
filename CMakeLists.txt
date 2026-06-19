cmake_minimum_required(VERSION 3.16)
project(larq_compute_engine C CXX)

include(CheckIncludeFile)

# Options and their default values
option(COMPILE_EXAMPLE "Enable compilation of the minimal example" ON)
option(COMPILE_BENCHMARK "Enable compilation of the benchmarking utility" ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LCE_SYSTEM_PROCESSOR_LOWER)
if(LCE_SYSTEM_PROCESSOR_LOWER MATCHES "^riscv" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    check_include_file("sys/hwprobe.h" LCE_HAS_SYS_HWPROBE_H)
endif()

if(LCE_SYSTEM_PROCESSOR_LOWER MATCHES "^riscv" AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT LCE_HAS_SYS_HWPROBE_H)
    message(STATUS "RISC-V Linux target without sys/hwprobe.h detected; using the portable fallback and disabling XNNPACK")
    set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "Disable XNNPACK when RISC-V cpuinfo hwprobe headers are unavailable" FORCE)
    # TensorFlow Lite's bundled ruy falls back to fixed cache parameters when
    # cpuinfo initialization fails. Provide a conservative shim so the build
    # does not depend on Linux riscv hwprobe headers.
    if(NOT TARGET cpuinfo)
        add_library(cpuinfo STATIC cmake/riscv_cpuinfo_shim/cpuinfo.c)
        target_include_directories(cpuinfo PUBLIC ${CMAKE_CURRENT_LIST_DIR}/cmake/riscv_cpuinfo_shim)
    endif()
    if(NOT TARGET cpuinfo::cpuinfo)
        add_library(cpuinfo::cpuinfo ALIAS cpuinfo)
    endif()
elseif(LCE_SYSTEM_PROCESSOR_LOWER MATCHES "^riscv" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "RISC-V Linux target with sys/hwprobe.h detected; keeping upstream cpuinfo and XNNPACK configuration")
endif()

# TensorFlow dependency, see https://www.tensorflow.org/lite/guide/build_cmake
set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/tensorflow")
set(TFLITE_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite")
add_subdirectory("${TFLITE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# Generic compilation options and settings
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_LIST_DIR})

# The LCE core files
set(LCE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/larq_compute_engine")
set(LCE_CORE_SRCS
        ${LCE_SOURCE_DIR}/tflite/kernels/bconv2d.cc
        ${LCE_SOURCE_DIR}/tflite/kernels/bmaxpool.cc
        ${LCE_SOURCE_DIR}/tflite/kernels/quantization.cc
        )
set(LCE_CORE_HDRS  # such that they can be discovered by IDEs such as CLion Visual Studio
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/kernel.h
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/kernel_4x2_portable.h
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/kernel_8x4x4_aarch64.h
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/kernel_8x4x1_aarch64.h
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/select_kernel.h
        ${LCE_SOURCE_DIR}/core/indirect_bgemm/kernel_8x4x2_aarch64.h
        ${LCE_SOURCE_DIR}/core/bmaxpool.h
        ${LCE_SOURCE_DIR}/core/bitpacking/utils.h
        ${LCE_SOURCE_DIR}/core/bitpacking/bitpack.h
        ${LCE_SOURCE_DIR}/core/bitpacking/bitpack_aarch64.h
        ${LCE_SOURCE_DIR}/core/types.h
        ${LCE_SOURCE_DIR}/core/bconv2d/optimized_indirect_bgemm.h
        ${LCE_SOURCE_DIR}/core/bconv2d/reference.h
        ${LCE_SOURCE_DIR}/core/bconv2d/optimized_bgemm.h
        ${LCE_SOURCE_DIR}/core/bconv2d/zero_padding_correction.h
        ${LCE_SOURCE_DIR}/core/bconv2d/params.h
        ${LCE_SOURCE_DIR}/core/bconv2d/output_transform.h
        ${LCE_SOURCE_DIR}/core/bgemm/kernels_common.h
        ${LCE_SOURCE_DIR}/core/bgemm/ruy_trmul_params.h
        ${LCE_SOURCE_DIR}/core/bgemm/kernels_aarch64.h
        ${LCE_SOURCE_DIR}/core/bgemm/kernels.h
        ${LCE_SOURCE_DIR}/core/bgemm/ruy_pack.h
        ${LCE_SOURCE_DIR}/core/bgemm/kernels_arm32.h
        ${LCE_SOURCE_DIR}/core/bgemm/bgemm.h
        ${LCE_SOURCE_DIR}/tflite/kernels/lce_ops_register.h
        ${LCE_SOURCE_DIR}/tflite/kernels/utils.h
        )

# The Larq-Compute-Engine version of the TFLite library (with BConvs)
add_library(larq-compute-engine ${LCE_CORE_SRCS} ${LCE_CORE_HDRS})
target_link_libraries(larq-compute-engine PUBLIC tensorflow-lite)

if (${CMAKE_SYSTEM_NAME} MATCHES "^(Darwin|Emscripten)$")  # macOS or WebAssembly
    target_link_libraries(larq-compute-engine PUBLIC dl)
else ()
    target_link_libraries(larq-compute-engine PUBLIC dl atomic)
endif ()

# The example application
if (COMPILE_EXAMPLE)
    set(LCE_EXAMPLE_SRCS ${CMAKE_CURRENT_LIST_DIR}/examples/lce_minimal.cc)
    add_executable(example ${LCE_EXAMPLE_SRCS})
    target_link_libraries(example larq-compute-engine)
endif ()

# The benchmarking binary
if (COMPILE_BENCHMARK)
    set(LCE_BENCHMARK_SRCS
            ${LCE_SOURCE_DIR}/tflite/benchmark/lce_benchmark_tflite_model.cc
            ${LCE_SOURCE_DIR}/tflite/benchmark/lce_benchmark_main.cc
            )
    set(LCE_BENCHMARK_HRDS
            ${LCE_SOURCE_DIR}/tflite/benchmark/lce_benchmark_tflite_model.h
            ${TFLITE_SOURCE_DIR}/tools/benchmark/benchmark_model.h
            )

    get_directory_property(TFLITE_BENCHMARK_SRCS DIRECTORY ${TFLITE_SOURCE_DIR}/tools/benchmark DEFINITION TFLITE_BENCHMARK_SRCS)
    list(FILTER TFLITE_BENCHMARK_SRCS EXCLUDE REGEX benchmark_main.cc)

    # The TSL dir is included in the tensorflow CMakeLists.txt but because we manually refer to those source files here we have to explicitly list this include directory again.
    set(TSL_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/third_party/xla/third_party/tsl")
    include_directories(
      ${TSL_SOURCE_DIR}
    )
    add_executable(lce_benchmark_model
            ${TFLITE_BENCHMARK_SRCS}
            ${LCE_BENCHMARK_SRCS} ${LCE_BENCHMARK_HRDS}
            )
    target_link_libraries(lce_benchmark_model larq-compute-engine)
endif ()
