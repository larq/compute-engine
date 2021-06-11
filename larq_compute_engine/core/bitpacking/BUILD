licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bitpack",
    hdrs = ["bitpack.h"] + select({
        "//larq_compute_engine:aarch64_build": [
            "bitpack_aarch64.h",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "bitpack_aarch64.h",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "bitpack_aarch64.h",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/core:types",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        ":bitpack",
    ],
)
