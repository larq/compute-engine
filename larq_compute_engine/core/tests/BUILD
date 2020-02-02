licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "bgemm_tests",
    size = "small",
    srcs = ["bgemm_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
        "//larq_compute_engine/core:bgemm_functor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "packbits_tests",
    size = "small",
    srcs = ["packbits_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
        "//larq_compute_engine/core:packbits",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fused_bgemm_tests",
    size = "small",
    srcs = ["fused_bgemm_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
        "//larq_compute_engine/core:fused_bgemm_functor",
        "//larq_compute_engine/core:utils",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "im2col_tests",
    size = "small",
    srcs = ["im2col_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
        "//larq_compute_engine/core:im2col_functor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bconv2d_tests",
    size = "small",
    srcs = ["bconv2d_tests.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//larq_compute_engine/core:bconv2d_functor",
        "@com_google_googletest//:gtest_main",
    ],
)

# collection of all CC architecture independent tests
# each new cc test needs to be added here
test_suite(
    name = "cc_tests_general",
    tests = [
        "bconv2d_tests",
        "bgemm_tests",
        "fused_bgemm_tests",
        "im2col_tests",
        "packbits_tests",
    ],
)

# CC tests for ARM32 architecture
cc_test(
    name = "packbits_arm32_tests",
    size = "small",
    srcs = ["packbits_arm32_tests.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//larq_compute_engine/core:packbits",
        "//larq_compute_engine/core:packbits_arm32",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "cc_tests_arm32",
    tests = [
        "packbits_arm32_tests",
    ],
)

# For running the ARM tests using qemu
# The script automatically runs every file that ends with _tests
sh_test(
    name = "cc_tests_arm32_qemu",
    srcs = ["cc_tests_arm32_qemu.sh"],
    data = [
        ":cc_tests_arm32",
        "@arm_compiler//:compiler_pieces",
    ],
)

cc_test(
    name = "packbits_aarch64_tests",
    size = "small",
    srcs = ["packbits_aarch64_tests.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//larq_compute_engine/core:packbits",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "cc_tests_aarch64",
    tests = [
        ":packbits_aarch64_tests",
        ":packbits_tests",
    ],
)

sh_test(
    name = "cc_tests_aarch64_qemu",
    srcs = ["cc_tests_aarch64_qemu.sh"],
    data = [
        ":cc_tests_aarch64",
        "@aarch64_compiler//:compiler_pieces",
    ],
)
