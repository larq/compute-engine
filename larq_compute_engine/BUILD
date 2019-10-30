licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = [
        "cc/utils/macros.h",
    ],
)

# BUILDING BGEMM
cc_library(
    name = "bgemm",
    hdrs = [
        "cc/core/bgemm_functor.h",
    ],
    deps = [
    ":utils",
    ]
)

cc_test(
    name = "bgemm_tests",
    size = "small",
    srcs = ["cc/tests/bgemm_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
         "@googletest//:gtest_main",
         ":bgemm",
    ],
)

# BUILDING PACKBITS
cc_library(
    name = "packbits",
    hdrs = [
        "cc/core/packbits.h",
    ],
)

cc_test(
    name = "packbits_tests",
    size = "small",
    srcs = ["cc/tests/packbits_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
         "@googletest//:gtest_main",
         ":packbits",
         ":utils",
    ],
)

cc_library(
    name = "fused_bgemm",
    hdrs = [
        "cc/core/fused_bgemm_functor.h",
    ],
    deps = [
         ":packbits",
         ":bgemm",
    ],
)

cc_test(
    name = "fused_bgemm_tests",
    size = "small",
    srcs = ["cc/tests/fused_bgemm_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
         "@googletest//:gtest_main",
         "@eigen_archive//:eigen",
         "fused_bgemm",
         ":utils",
    ],
)

# BUILDING IM2COL
cc_library(
    name = "im2col",
    hdrs = [
        "cc/core/im2col_functor.h",
    ],
)

cc_test(
    name = "im2col_tests",
    size = "small",
    srcs = ["cc/tests/im2col_tests.cc"],
    copts = ["-Iext/gtest/include"],
    deps = [
         "@googletest//:gtest_main",
         ":im2col",
    ],
)

cc_library(
    name = "padding",
    hdrs = ["cc/core/padding_functor.h"],
)

cc_library(
    name = "bconv2d",
    hdrs = [
        "cc/core/bconv2d_functor.h",
    ],
    deps = [
         ":fused_bgemm",
         ":padding",
    ],
)

cc_test(
    name = "bconv2d_tests",
    size = "small",
    srcs = ["cc/tests/bconv2d_tests.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
         "@googletest//:gtest_main",
         ":bconv2d",
    ],
)

# COLLECTION OF ALL CC TESTS
# each new cc test needs to be added here
test_suite(
    name = "cc_tests",
    tests = [
        "bgemm_tests",
        "packbits_tests",
        "fused_bgemm_tests",
        "im2col_tests",
        "bconv2d_tests",
    ],
)

# MAIN COMPUTE ENGINE LIBRARY
cc_binary(
    name = 'python/ops/_larq_compute_engine_ops.so',
    srcs = [
        "cc/kernels/sign_kernels.cc",
        "cc/kernels/bconv2d.cc",
        "cc/ops/compute_engine_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        ":utils",
        ":bconv2d",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

py_runtime(
    name = "pyruntime",
    interpreter_path = select({
        # Update paths as appropriate for your system.
        "@bazel_tools//tools/python:PY2": "/usr/bin/python2",
        "@bazel_tools//tools/python:PY3": "/usr/bin/python3",
    }),
    files = [],
)

py_library(
    name = "compute_engine_ops_py",
    srcs = ([
        "python/ops/compute_engine_ops.py",
    ]),
    data = [
        ":python/ops/_larq_compute_engine_ops.so"
    ],
    srcs_version = "PY3",
)

py_library(
    name = "compute_engine_utils_py",
    srcs = ([
        "python/utils.py",
        "python/modelconverter.py",
    ]),
    srcs_version = "PY3",
)

py_library(
    name = "compute_engine_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":compute_engine_ops_py",
        ":compute_engine_utils_py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "bsign_ops_py_test",
    srcs = [
        "python/ops/bsign_ops_test.py"
    ],
    main = "python/ops/bsign_ops_test.py",
    deps = [
        ":compute_engine_ops_py",
        ":compute_engine_utils_py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_test(
    name = "bconv2d_ops_py_test",
    srcs = [
        "python/ops/bconv2d_ops_test.py"
    ],
    main = "python/ops/bconv2d_ops_test.py",
    deps = [
        ":compute_engine_ops_py",
        ":compute_engine_utils_py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

# collection of all python tests
test_suite(
    name = "py_tests",
    tests = [
        "bsign_ops_py_test",
        "bconv2d_ops_py_test",
    ],
)
