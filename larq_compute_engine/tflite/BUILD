load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bgemm_ref",
    hdrs = [
        "cc/kernels/bgemm_impl_ref.h",
    ],
    deps = [
        "//larq_compute_engine:bgemm",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
    ]
)

cc_library(
    name = "bgemm_kernels_ruy",
    hdrs = [
        "cc/kernels/bgemm_kernels_arm.h",
        "cc/kernels/bgemm_kernels_ruy.h",
        "cc/kernels/bgemm_kernels_x86.h",
    ],
    deps = [
        "//larq_compute_engine:bgemm",
    ]
)

cc_library(
    name = "bgemm_ruy",
    hdrs = [
        "cc/kernels/bgemm_impl_ruy.h",
        "cc/kernels/bgemm_trmul_params.h",
    ],
    deps = [
        ":bgemm_kernels_ruy",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
    ]
)

cc_library(
    name = "bgemm",
    hdrs = [
        "cc/kernels/bgemm_impl.h",
    ],
    deps = [
        ":bgemm_ref",
        ":bgemm_ruy",
    ]
)

cc_library(
    name = "bconv2d_impl",
    hdrs = [
        "cc/kernels/bconv2d_impl.h",
    ],
    deps = [
        ":bgemm",
        "//larq_compute_engine:bconv2d",
        "//larq_compute_engine:padding",
        "//larq_compute_engine:packbits",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_gemm",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
    ]
)

cc_library(
    name = "bconv2d",
    srcs = ["cc/kernels/bconv2d.cc"],
    deps = [
        ":bconv2d_impl",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:context",
        "@org_tensorflow//tensorflow/lite/kernels/internal:kernel_utils",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@gemmlowp//:profiler",
    ],
)

cc_test(
    name = "bconv2d_test",
    size = "small",
    srcs = ["tests/bconv2d_test.cc"],
    deps = [
        ":bconv2d",
        "@org_tensorflow//tensorflow/lite/kernels:test_main",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
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

# The script automatically runs every file that ends with _test
sh_test(
    name = "cc_tests_arm32_qemu",
    srcs = ["cc_tests_arm32_qemu.sh"],
    data = [
        ":cc_tests",
        "@arm_compiler//:compiler_pieces",
    ],
)

# The script automatically runs every file that ends with _test
sh_test(
    name = "cc_tests_aarch64_qemu",
    srcs = ["cc_tests_aarch64_qemu.sh"],
    data = [
        ":cc_tests",
        "@aarch64_compiler//:compiler_pieces",
    ],
)
