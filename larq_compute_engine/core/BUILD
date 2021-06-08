licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels/internal:cppmath",
    ],
)

cc_library(
    name = "bmaxpool",
    hdrs = [
        "bmaxpool.h",
    ],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)
