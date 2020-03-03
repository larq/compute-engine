load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")
load("//larq_compute_engine/tflite:build_defs.bzl", "ruy_copts_base")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bconv2d_impl",
    hdrs = [
        "bconv2d_impl.h",
    ],
    deps = [
        "//larq_compute_engine/core:bgemm_impl",
        "//larq_compute_engine/core:packbits",
        "//larq_compute_engine/core:padding_functor",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
    ],
)

cc_library(
    name = "lce_op_kernels",
    srcs = [
        "bconv2d.cc",
        "sign.cc",
    ],
    hdrs = [
        "lce_ops_register.h",
        "utils.h",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    deps = [
        ":bconv2d_impl",
        "@flatbuffers",
        "@gemmlowp//:profiler",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:c_api_internal",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
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
