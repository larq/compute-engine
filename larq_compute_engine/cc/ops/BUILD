licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bconv_shape_inference",
    srcs = ["bconv_shape_inference.cc"],
    hdrs = ["bconv_shape_inference.h"],
    deps = ["@local_config_tf//:tf_header_lib"],
)

cc_library(
    name = "compute_engine_ops",
    srcs = ["compute_engine_ops.cc"],
    deps = [
        "@local_config_tf//:tf_header_lib",
        ":bconv_shape_inference",
    ]
)
