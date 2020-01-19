licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# BUILDING BGEMM
cc_library(
    name = "bgemm",
    hdrs = [
        "bgemm_functor.h",
    ],
    deps = [
    "//larq_compute_engine/cc/utils:utils",
    ]
)

# BUILDING PACKBITS
cc_library(
    name = "packbits",
    hdrs = ["packbits.h"] + select({
        "//larq_compute_engine:aarch64_build": [
            "packbits_aarch64.h",
        ],
        "//larq_compute_engine:arm32_build": [
            "packbits_arm32.h",
        ],
        "@org_tensorflow//tensorflow:android_arm": [
            "packbits_arm32.h",
        ],
        "@org_tensorflow//tensorflow:android_arm64": [
            "packbits_aarch64.h",
        ],
        "//conditions:default": [],
    }),
    deps = ["//larq_compute_engine/cc/utils:utils"],
)

cc_library(
    name = "fused_bgemm",
    hdrs = [
        "fused_bgemm_functor.h",
    ],
    deps = [
         ":packbits",
         ":bgemm",
    ],
)

# BUILDING IM2COL
cc_library(
    name = "im2col",
    hdrs = [
        "im2col_functor.h",
    ],
)

cc_library(
    name = "padding",
    hdrs = ["padding_functor.h"],
)

cc_library(
    name = "bconv2d",
    hdrs = [
        "bconv2d_functor.h",
    ],
    deps = [
         ":fused_bgemm",
         ":padding",
    ],
)

cc_library(
    name = "packbits_arm32",
    hdrs = [
        "packbits_arm32.h",
    ],
)

cc_library(
    name = "packbits_aarch64",
    hdrs = [
        "packbits_aarch64.h",
    ],
)
