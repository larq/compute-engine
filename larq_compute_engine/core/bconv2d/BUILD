licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "params",
    hdrs = [
        "params.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
    ],
)

cc_library(
    name = "output_transform",
    hdrs = [
        "output_transform.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:cppmath",
    ],
)

cc_library(
    name = "zero_padding_correction",
    hdrs = ["zero_padding_correction.h"],
)

cc_library(
    name = "reference",
    hdrs = [
        "reference.h",
    ],
    deps = [
        ":output_transform",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "optimized_bgemm",
    hdrs = [
        "optimized_bgemm.h",
    ],
    deps = [
        ":zero_padding_correction",
        "//larq_compute_engine/core/bgemm",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "optimized_indirect_bgemm",
    hdrs = [
        "optimized_indirect_bgemm.h",
    ],
    deps = [
        ":zero_padding_correction",
        "//larq_compute_engine/core/indirect_bgemm:kernels",
        "//larq_compute_engine/core/indirect_bgemm:prepare",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
        "@ruy//ruy/profiler:instrumentation",
    ],
)
