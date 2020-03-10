load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_end2end_verify",
    srcs = ["end2end_verify.cc"],
    linkopts = tflite_linkopts(),
    module_name = "end2end_verify",
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@pybind11",
    ],
)

py_test(
    name = "end2end_test",
    srcs = ["end2end_test.py"],
    deps = [
        ":_end2end_verify",
        "//larq_compute_engine/mlir:converter",
    ],
)
