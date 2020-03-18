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

# collection of all CC architecture independent tests
# each new cc test needs to be added here
test_suite(
    name = "cc_tests",
    tests = [
        "bgemm_tests",
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
