load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()

cc_binary(
    name = "lce_benchmark_model",
    srcs = [
        "lce_benchmark_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "@org_tensorflow//tensorflow/lite/tools/benchmark:logging",
    ],
)

cc_binary(
    name = "lce_benchmark_all",
    srcs = [
        "lce_benchmark_all.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@googlebenchmark//:benchmark",
        "@googlebenchmark//:benchmark_main",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
