load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")
load("//larq_compute_engine/tflite:build_defs.bzl", "ruy_copts_base")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bconv2d_params",
    hdrs = [
        "bconv2d_params.h",
    ],
)

cc_library(
    name = "bconv2d_output_transform_utils",
    hdrs = [
        "bconv2d_output_transform_utils.h",
    ],
    deps = [
        ":bconv2d_params",
        "//larq_compute_engine/core:types",
        "//larq_compute_engine/core/bconv2d:output_transform",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "lce_op_kernels",
    srcs = [
        "bconv2d.cc",
        "bmaxpool.cc",
        "quantization.cc",
    ],
    hdrs = [
        "lce_ops_register.h",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    deps = [
        ":bconv2d_output_transform_utils",
        ":bconv2d_params",
        ":utils",
        "//larq_compute_engine/core:bmaxpool",
        "//larq_compute_engine/core/bconv2d:optimized",
        "//larq_compute_engine/core/bconv2d:reference",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "//larq_compute_engine/core/bitpacking:utils",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@ruy//ruy/profiler:instrumentation",
    ],
    alwayslink = 1,
)
