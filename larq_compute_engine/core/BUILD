licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
)

cc_library(
    name = "bitpack",
    hdrs = ["bitpack.h"] + select({
        "//larq_compute_engine:aarch64_build": [
            "bitpack_aarch64.h",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "bitpack_aarch64.h",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":types",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bitpack_utils",
    hdrs = ["bitpack_utils.h"],
    deps = [
        ":bitpack",
    ],
)

cc_library(
    name = "padding_functor",
    hdrs = ["padding_functor.h"],
)

cc_library(
    name = "bconv2d_output_transform",
    hdrs = [
        "bconv2d_output_transform.h",
    ],
    deps = [
        ":types",
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
    ],
    deps = [
        ":bgemm_kernels_arm",
        ":bitpack",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "bgemm_impl",
    hdrs = [
        "bgemm_impl.h",
        "bgemm_trmul_params.h",
        "ruy_pack.h",
    ],
    deps = [
        ":bgemm_kernels_ruy",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
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
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "bmaxpool",
    hdrs = [
        "bmaxpool.h",
    ],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)
