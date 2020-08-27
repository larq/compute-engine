package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "bconv2d_op_model",
    hdrs = [
        "bconv2d_op_model.h",
    ],
    deps = [
        ":utils",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@flatbuffers",
    ],
)

cc_test(
    name = "bconv2d_test",
    size = "large",
    srcs = ["bconv2d_test.cc"],
    deps = [
        ":bconv2d_op_model",
        ":utils",
        "//larq_compute_engine/core:bitpack",
        "//larq_compute_engine/core:bitpack_utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

cc_test(
    name = "bconv2d_int8_test",
    size = "large",
    srcs = ["bconv2d_int8_test.cc"],
    deps = [
        ":bconv2d_op_model",
        ":utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

cc_test(
    name = "bmaxpool_test",
    size = "small",
    srcs = ["bmaxpool_test.cc"],
    deps = [
        ":utils",
        "//larq_compute_engine/core:bitpack_utils",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

cc_test(
    name = "quantization_test",
    size = "small",
    srcs = ["quantization_test.cc"],
    deps = [
        ":utils",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
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
        ":bconv2d_int8_test",
        ":bconv2d_test",
        ":bmaxpool_test",
        ":quantization_test",
    ],
)
