package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "bconv2d_test",
    size = "small",
    srcs = ["bconv2d_test.cc"],
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_ops",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

# COLLECTION OF ALL TFLITE CC TESTS
# each new cc test needs to be added here
test_suite(
    name = "cc_tests",
    tests = [
        ":bconv2d_test",
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
