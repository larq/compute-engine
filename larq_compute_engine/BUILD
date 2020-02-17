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

# dynamic library of larq compute engine custom ops
# for tensorflow
cc_binary(
    name = "tf/python/ops/_larq_compute_engine_ops.so",
    srcs = [
        "tf/kernels/bconv2d.cc",
        "tf/kernels/sign_kernels.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "//larq_compute_engine/core:bconv2d_functor",
        "//larq_compute_engine/core:utils",
        "//larq_compute_engine/tf/ops:compute_engine_ops",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_runtime(
    name = "pyruntime",
    files = [],
    interpreter_path = select({
        # Update paths as appropriate for your system.
        "@bazel_tools//tools/python:PY2": "/usr/bin/python2",
        "@bazel_tools//tools/python:PY3": "/usr/bin/python3",
    }),
)

py_library(
    name = "compute_engine_ops_py",
    srcs = ([
        "tf/python/ops/compute_engine_ops.py",
    ]),
    data = [
        ":tf/python/ops/_larq_compute_engine_ops.so",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "compute_engine_utils_py",
    srcs = ([
        "tf/python/utils.py",
        "tf/python/modelconverter.py",
    ]),
    srcs_version = "PY3",
)

py_library(
    name = "compute_engine_py",
    srcs = ([
        "__init__.py",
        "//larq_compute_engine/mlir:__init__.py",
        "//larq_compute_engine/mlir:python/__init__.py",
    ]),
    deps = [
        "//larq_compute_engine/mlir:converter",
    ],
)

py_test(
    name = "bsign_ops_py_test",
    size = "small",
    srcs = [
        "tf/python/ops/bsign_ops_test.py",
    ],
    main = "tf/python/ops/bsign_ops_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compute_engine_ops_py",
        ":compute_engine_utils_py",
    ],
)

py_test(
    name = "bconv2d_ops_py_test",
    size = "large",
    srcs = [
        "tf/python/ops/bconv2d_ops_test.py",
    ],
    main = "tf/python/ops/bconv2d_ops_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compute_engine_ops_py",
        ":compute_engine_utils_py",
    ],
)

# collection of all python tests
test_suite(
    name = "py_tests",
    tests = [
        "bconv2d_ops_py_test",
        "bsign_ops_py_test",
    ],
)


sh_binary(
    name = "download_compiler",
    srcs = ["download_compiler_placeholder.sh"],
    data = select({
        ":arm32_build": ["@arm_compiler//:compiler_pieces"],
        ":aarch64_build": ["@aarch64_compiler//:compiler_pieces"],
        "//conditions:default": [],
    }),
)
