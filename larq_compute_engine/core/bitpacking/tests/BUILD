licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "bitpack_test",
    size = "small",
    srcs = ["bitpack_test.cc"],
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
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# Collection of all CC architecture independent tests. Each new cc test needs to
# be added here.
test_suite(
    name = "cc_tests",
    tests = [
        "bitpack_test",
    ],
)

cc_test(
    name = "bitpack_aarch64_test",
    size = "small",
    srcs = ["bitpack_aarch64_test.cc"],
    deps = [
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "cc_tests_aarch64",
    tests = [
        ":bitpack_aarch64_test",
        ":bitpack_test",
    ],
)
