licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "benchmark_aarch64",
    linkopts = ["-lrt -lm"],
    srcs = [
        "aarch64/packbits_bench.cc",
    ],
    deps = [
        "@googlebenchmark//:benchmark",
        "@googlebenchmark//:benchmark_main",
        "//larq_compute_engine/cc/core:packbits",
        "//larq_compute_engine/cc/core:packbits_aarch64",
    ],
)

cc_binary(
        name = "benchmark",
        linkopts = ["-lrt -lm"],
    srcs = [
        "bconv2d_bench.cc",
        "bgemm_bench.cc",
        "im2col_bench.cc",
        "padding_bench.cc",
    ],
        deps = [
        "@googlebenchmark//:benchmark",
        "@googlebenchmark//:benchmark_main",
        "//larq_compute_engine/cc/core:bconv2d",
        "//larq_compute_engine/cc/core:bgemm",
        "//larq_compute_engine/cc/core:fused_bgemm",
        "//larq_compute_engine/cc/core:im2col",
        "//larq_compute_engine/cc/core:padding",
        ],
)
