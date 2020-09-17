load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "interpreter_wrapper_utils",
    hdrs = ["interpreter_wrapper_utils.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@pybind11",
    ],
)

pybind_extension(
    name = "interpreter_wrapper_lite",
    srcs = ["interpreter_wrapper_lite.cc"],
    linkopts = tflite_linkopts(),
    module_name = "interpreter_wrapper_lite",
    deps = [
        ":interpreter_wrapper_utils",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@pybind11",
    ],
)

py_library(
    name = "interpreter",
    srcs = [
        "__init__.py",
        "interpreter.py",
    ],
    deps = [
        ":interpreter_wrapper_lite",
    ],
)
