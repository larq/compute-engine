load("//larq_compute_engine/mlir/tests:lit_test.bzl", "lce_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

lce_lit_test_suite(
    name = "lit",
    srcs = glob(["*.mlir"]),
    data = [
        "//larq_compute_engine/mlir:lce-tf-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

cc_test(
    name = "lce_ops_options_test",
    srcs = ["lce_ops_options_test.cc"],
    deps = [
        "//larq_compute_engine/mlir:larq_compute_engine",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)
