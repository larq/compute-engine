load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "lce_benchmark_model",
    srcs = [
        "lce_benchmark_main.cc",
    ],
    copts = ["-Wall"] + tflite_copts(),
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite/tools:logging",
        "@org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)
