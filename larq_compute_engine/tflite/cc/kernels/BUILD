load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts", "tflite_copts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")
load("//larq_compute_engine/tflite:build_defs.bzl", "ruy_copts_base")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bgemm_ref",
    hdrs = [
        "bgemm_impl_ref.h",
    ],
    deps = [
        "//larq_compute_engine:bgemm",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
    ]
)

cc_library(
    name = "bgemm_kernels_common",
    hdrs = [
        "bgemm_kernels_common.h",
    ],
)

cc_library(
    name = "bgemm_kernels_arm",
    hdrs = [
        "bgemm_kernels_arm.h",
        "bgemm_kernels_arm64.h",
    ],
    deps = [
        ":bgemm_kernels_common"
    ]
)

cc_library(
    name = "bgemm_kernels_ruy",
    hdrs = [
        "bgemm_kernels_arm.h",
        "bgemm_kernels_ruy.h",
        "bgemm_kernels_x86.h",
    ],
    deps = [
        "//larq_compute_engine:bgemm",
        ":bgemm_kernels_arm"
    ]
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
    ]
)

cc_library(
    name = "bgemm",
    hdrs = [
        "bgemm_impl.h",
    ],
    deps = [
        ":bgemm_ref",
        ":bgemm_ruy",
    ]
)

cc_library(
    name = "bconv2d_impl",
    hdrs = [
        "bconv2d_impl.h",
    ],
    deps = [
        ":bgemm",
        "//larq_compute_engine:bconv2d",
        "//larq_compute_engine:padding",
        "//larq_compute_engine:packbits",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
    ]
)

cc_library(
    name = "lce_op_kernels",
    srcs = [
        "sign.cc",
        "bconv2d.cc",
    ],
    hdrs = ["utils.h"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    visibility = ["//visibility:private"],
    deps = [
        ":bconv2d_impl",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
        "@gemmlowp//:profiler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lce_ops",
    srcs = ["lce_register.cc"],
     deps = [
        ":lce_op_kernels",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)
