licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = [
        "macros.h",
        "types.h",
    ],
)

# BUILDING BGEMM
cc_library(
    name = "bgemm_functor",
    hdrs = [
        "bgemm_functor.h",
    ],
    deps = [
        ":utils",
    ],
)

# BUILDING PACKBITS
cc_library(
    name = "packbits",
    hdrs = ["packbits.h"] + select({
        "//larq_compute_engine:aarch64_build": [
            "packbits_aarch64.h",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "packbits_aarch64.h",
        ],
        "//conditions:default": [],
    }),
    deps = [":utils"],
)

cc_library(
    name = "packbits_utils",
    hdrs = ["packbits_utils.h"],
    deps = [
        ":packbits",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "padding_functor",
    hdrs = ["padding_functor.h"],
)

cc_library(
    name = "packbits_aarch64",
    hdrs = [
        "packbits_aarch64.h",
    ],
)

cc_library(
    name = "bgemm_ref",
    hdrs = [
        "bgemm_impl_ref.h",
    ],
    deps = [
        "//larq_compute_engine/core:bgemm_functor",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bconv2d_output_transform",
    hdrs = [
        "bconv2d_output_transform.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:cppmath",
    ],
)

cc_library(
    name = "bgemm_kernels_common",
    hdrs = [
        "bgemm_kernels_common.h",
    ],
    deps = [
        ":bconv2d_output_transform",
    ],
)

cc_library(
    name = "bgemm_kernels_arm",
    hdrs = [
        "bgemm_kernels_arm.h",
        "bgemm_kernels_arm32.h",
        "bgemm_kernels_arm64.h",
    ],
    deps = [
        ":bgemm_kernels_common",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bgemm_kernels_ruy",
    hdrs = [
        "bgemm_kernels_arm.h",
        "bgemm_kernels_ruy.h",
        "bgemm_kernels_x86.h",
    ],
    deps = [
        ":bgemm_kernels_arm",
        "//larq_compute_engine/core:bgemm_functor",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bgemm_ruy",
    hdrs = [
        "bgemm_impl_ruy.h",
        "bgemm_trmul_params.h",
    ],
    deps = [
        ":bgemm_kernels_ruy",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bgemm_impl",
    hdrs = [
        "bgemm_impl.h",
    ],
    deps = [
        ":bgemm_ref",
        ":bgemm_ruy",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bconv2d_impl_ref",
    hdrs = [
        "bconv2d_impl_ref.h",
    ],
    deps = [
        ":bconv2d_output_transform",
        ":bgemm_functor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "bmaxpool",
    hdrs = [
        "bmaxpool.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)
