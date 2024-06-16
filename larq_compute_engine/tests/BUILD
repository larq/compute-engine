load("@pypi//:requirements.bzl", tf_requirement = "requirement")
load("@pypi_lce//:requirements.bzl", lce_requirement = "requirement")
load("//larq_compute_engine/tests:qemu_test.bzl", "lce_qemu_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "test_arm32_binary.sh",
    "test_aarch64_binary.sh",
])

py_test(
    name = "end2end_test",
    size = "large",
    srcs = ["end2end_test.py"],
    deps = [
        "//larq_compute_engine/mlir:converter",
        "//larq_compute_engine/tflite/python:interpreter",
        tf_requirement("numpy"),
        lce_requirement("larq"),
        lce_requirement("pytest"),
        lce_requirement("tensorflow"),
        lce_requirement("tensorflow_datasets"),
        lce_requirement("tf-keras"),
    ],
)

py_test(
    name = "strip_lcedequantize_test",
    srcs = ["strip_lcedequantize_test.py"],
    deps = [
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
        "//larq_compute_engine/core/bitpacking/tests:cc_tests",
        "//larq_compute_engine/tflite/tests:cc_tests",
    ],
)

lce_qemu_test_suite(
    name = "arm32_tests",
    platform = "arm32",
    tests = [
        "//larq_compute_engine/tflite/tests:bconv2d_test",
        "//larq_compute_engine/tflite/tests:bmaxpool_test",
        "//larq_compute_engine/tflite/tests:quantization_test",
    ],
)

lce_qemu_test_suite(
    name = "aarch64_tests",
    platform = "aarch64",
    tests = [
        "//larq_compute_engine/core/bitpacking/tests:bitpack_aarch64_test",
        "//larq_compute_engine/core/bitpacking/tests:bitpack_test",
        "//larq_compute_engine/tflite/tests:bconv2d_test",
        "//larq_compute_engine/tflite/tests:bmaxpool_test",
        "//larq_compute_engine/tflite/tests:quantization_test",
    ],
)
