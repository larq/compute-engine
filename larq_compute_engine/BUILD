licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "arm32_build",
    values = {"cpu": "armeabi"},
)
config_setting(
    name = "aarch64_build",
    values = {"cpu": "aarch64"},
)

cc_library(
    name = "bconv_shape_inference",
    srcs = ["cc/ops/bconv_shape_inference.cc"],
    hdrs = ["cc/ops/bconv_shape_inference.h"],
    deps = ["@local_config_tf//:tf_header_lib"],
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
        "//larq_compute_engine/cc/utils:utils",
        "//larq_compute_engine/cc/core:bconv2d",
        ":bconv_shape_inference",
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
    size = "small",
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
    size = "large",
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
