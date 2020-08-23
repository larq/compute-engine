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
        "//larq_compute_engine/core:padding_functor",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
        "@ruy//ruy/profiler:instrumentation",
    ],
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
        "//larq_compute_engine/core:bconv2d_output_transform",
        "//larq_compute_engine/core:types",
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
        ":bconv2d_impl",
        ":bconv2d_output_transform_utils",
        ":bconv2d_params",
        ":utils",
        "//larq_compute_engine/core:bconv2d_impl_ref",
        "//larq_compute_engine/core:bitpack",
        "//larq_compute_engine/core:bitpack_utils",
        "//larq_compute_engine/core:bmaxpool",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@ruy//ruy/profiler:instrumentation",
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
