load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "@org_tensorflow//tensorflow:android": [
        "-pie",
        "-llog",
    ],
})

cc_library(
    name = "imagenet_model_evaluator",
    srcs = ["imagenet_model_evaluator.cc"],
    hdrs = ["imagenet_model_evaluator.h"],
    copts = tflite_copts(),
    deps = [
        "//larq_compute_engine/tflite/tools/evaluation/stages:image_classification_stage",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/lite/c:c_api_internal",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_binary(
    name = "imagenet_accuracy_eval",
    srcs = ["imagenet_accuracy_eval.cc"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        ":imagenet_model_evaluator",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/lite/c:c_api_internal",
        "@org_tensorflow//tensorflow/lite/profiling:time",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/lite/tools/accuracy:csv_writer",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)
