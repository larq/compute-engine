load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_cc_binary(
    name = "lce_benchmark_model",
    srcs = [
        "lce_benchmark_main.cc",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/benchmark:lce_benchmark_tflite_model_lib",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite/tools:logging",
    ],
)

cc_library(
    name = "lce_benchmark_tflite_model_lib",
    srcs = ["lce_benchmark_tflite_model.cc"],
    hdrs = ["lce_benchmark_tflite_model.h"],
    copts = tflite_copts() + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)
