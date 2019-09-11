licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_larq_compute_engine_ops.so',
    srcs = [
        "cc/kernels/compute_engine_kernels.cc",
        "cc/kernels/sign_kernels.cc",
        "cc/ops/compute_engine_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

cc_test(
    name = "bgemm_tests",
    size = "small",
    srcs = ["cc/tests/bgemm_tests.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = ["@gtest//:main"],
)

py_library(
    name = "compute_engine_ops_py",
    srcs = ([
        "python/ops/compute_engine_ops.py",
    ]),
    data = [
        ":python/ops/_larq_compute_engine_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "compute_engine_ops_py_test",
    srcs = [
        "python/ops/compute_engine_ops_test.py"
    ],
    main = "python/ops/compute_engine_ops_test.py",
    deps = [
        ":compute_engine_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "compute_engine_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":compute_engine_ops_py"
    ],
    srcs_version = "PY2AND3",
)
