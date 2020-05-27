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
    size = "large",
    srcs = ["end2end_test.py"],
    deps = [
        ":_end2end_verify",
        "//larq_compute_engine/mlir:converter",
    ],
)

py_test(
    name = "convert_model",
    srcs = ["convert_model.py"],
    deps = [
        "//larq_compute_engine/mlir:converter",
    ],
)

test_suite(
    name = "cc_tests",
    tests = [
        "//larq_compute_engine/core/tests:cc_tests",
        "//larq_compute_engine/tflite/tests:cc_tests",
    ],
)

test_suite(
    name = "cc_tests_arm32",
    tests = [
        "//larq_compute_engine/core/tests:cc_tests_arm32",
        "//larq_compute_engine/tflite/tests:cc_tests",
    ],
)

sh_test(
    name = "cc_tests_arm32_qemu",
    size = "large",
    srcs = ["cc_tests_arm32_qemu.sh"],
    data = [
        ":cc_tests_arm32",
        "@arm_compiler//:compiler_pieces",
    ],
)

test_suite(
    name = "cc_tests_aarch64",
    tests = [
        "//larq_compute_engine/core/tests:cc_tests_aarch64",
        "//larq_compute_engine/tflite/tests:cc_tests",
    ],
)

sh_test(
    name = "cc_tests_aarch64_qemu",
    size = "large",
    srcs = ["cc_tests_aarch64_qemu.sh"],
    data = [
        ":cc_tests_aarch64",
        "@aarch64_compiler//:aarch64_compiler_pieces",
    ],
)
