package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "bconv2d_test",
    size = "medium",
    srcs = ["bconv2d_test.cc"],
    deps = [
        "//larq_compute_engine/core:packbits",
        "//larq_compute_engine/core:packbits_utils",
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
