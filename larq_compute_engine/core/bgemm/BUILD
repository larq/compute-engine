licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kernels_common",
    hdrs = [
        "kernels_common.h",
    ],
    deps = [
        "//larq_compute_engine/core/bconv2d:output_transform",
    ],
)

cc_library(
    name = "kernels",
    hdrs = [
        "kernels.h",
        "kernels_aarch64.h",
        "kernels_arm32.h",
    ],
    deps = [
        ":kernels_common",
        "//larq_compute_engine/core/bitpacking:utils",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bgemm",
    hdrs = [
        "bgemm.h",
        "ruy_pack.h",
        "ruy_trmul_params.h",
    ],
    deps = [
        ":kernels",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy/profiler:instrumentation",
    ],
)
