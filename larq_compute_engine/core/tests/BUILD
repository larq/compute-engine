licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "bitpack_tests",
    size = "small",
    srcs = ["bitpack_tests.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "@org_tensorflow//tensorflow:macos": [
            "-lm",
        ],
        "//conditions:default": [
            "-lm",
            "-lrt",
        ],
    }),
    deps = [
        "//larq_compute_engine/core:bitpack",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# collection of all CC architecture independent tests
# each new cc test needs to be added here
test_suite(
    name = "cc_tests",
    tests = [
        "bitpack_tests",
    ],
)

cc_test(
    name = "bitpack_aarch64_tests",
    size = "small",
    srcs = ["bitpack_aarch64_tests.cc"],
    deps = [
        "//larq_compute_engine/core:bitpack",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "cc_tests_aarch64",
    tests = [
        ":bitpack_aarch64_tests",
        ":bitpack_tests",
    ],
)
