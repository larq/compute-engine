licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "prepare",
    hdrs = [
        "prepare.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "//larq_compute_engine/tflite/kernels:bconv2d_params",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "kernels",
    hdrs = [
        "kernel.h",
        "kernel_4x2_portable.h",
        "kernel_8x4x1_aarch64.h",
        "kernel_8x4x2_aarch64.h",
        "kernel_8x4x4_aarch64.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "//larq_compute_engine/core/bconv2d:output_transform",
        "//larq_compute_engine/tflite/kernels:bconv2d_params",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@ruy//ruy/profiler:instrumentation",
    ],
)
