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
