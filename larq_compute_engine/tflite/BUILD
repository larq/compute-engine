load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load(":build_defs.bzl", "ruy_copts_base")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# COLLECTION OF ALL TFLITE CC TESTS
# each new cc test needs to be added here
test_suite(
    name = "cc_tests",
    tests = [
        "//larq_compute_engine/tflite/tests:bconv2d_test",
    ],
)

# The script automatically runs every file that ends with _test
sh_test(
    name = "cc_tests_arm32_qemu",
    srcs = ["cc_tests_arm32_qemu.sh"],
    data = [
        ":cc_tests",
        "@arm_compiler//:compiler_pieces",
    ],
)

# The script automatically runs every file that ends with _test
sh_test(
    name = "cc_tests_aarch64_qemu",
    srcs = ["cc_tests_aarch64_qemu.sh"],
    data = [
        ":cc_tests",
        "@aarch64_compiler//:compiler_pieces",
    ],
)
