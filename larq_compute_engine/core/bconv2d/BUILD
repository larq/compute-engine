licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

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
    name = "padding_functor",
    hdrs = ["padding_functor.h"],
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
    name = "optimized",
    hdrs = [
        "optimized.h",
    ],
    deps = [
        ":padding_functor",
        "//larq_compute_engine/core/bgemm",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
        "@ruy//ruy/profiler:instrumentation",
    ],
)
