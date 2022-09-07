cmake_minimum_required(VERSION 3.16)
project(larq_compute_engine C CXX)

# Options and their default values
option(COMPILE_EXAMPLE "Enable compilation of the minimal example" ON)
option(COMPILE_BENCHMARK "Enable compilation of the benchmarking utility" ON)

# TensorFlow dependency, see https://www.tensorflow.org/lite/guide/build_cmake
set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/tensorflow/")
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

# The example application
if(COMPILE_EXAMPLE)
    set(LCE_EXAMPLE_SRCS ${CMAKE_CURRENT_LIST_DIR}/examples/lce_minimal.cc)
    add_executable(example ${LCE_CORE_SRCS} ${LCE_CORE_HDRS} ${LCE_EXAMPLE_SRCS})
    target_link_libraries(example tensorflow-lite)
endif()

# The benchmarking binary
if(COMPILE_BENCHMARK)
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

    add_executable(lce_benchmark_model
            ${TFLITE_BENCHMARK_SRCS}
            ${LCE_CORE_SRCS} ${LCE_CORE_HDRS}
            ${LCE_BENCHMARK_SRCS} ${LCE_BENCHMARK_HRDS}
            )
    target_link_libraries(lce_benchmark_model tensorflow-lite)
endif()
