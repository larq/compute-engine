package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "bconv2d_test",
    size = "small",
    srcs = ["bconv2d_test.cc"],
    deps = [
        "//larq_compute_engine/tflite/cc/kernels:lce_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
