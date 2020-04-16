package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
)

cc_library(
    name = "bconv2d_op",
    hdrs = [
        "bconv2d_op.h",
    ],
    deps = [
        ":utils",
        "//larq_compute_engine/tflite/kernels:lce_ops",
        "@flatbuffers",
    ],
)

cc_test(
    name = "bconv2d_test",
    size = "large",
    srcs = ["bconv2d_test.cc"],
    deps = [
        ":bconv2d_op",
        ":utils",
        "//larq_compute_engine/core:packbits",
        "//larq_compute_engine/core:packbits_utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

cc_test(
    name = "bconv2d_int8",
    size = "large",
    srcs = ["bconv2d_int8.cc"],
    deps = [
        ":bconv2d_op",
        ":utils",
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
        ":bconv2d_int8",
        ":bconv2d_test",
    ],
)
