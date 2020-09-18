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

py_library(
    name = "compute_engine_py",
    srcs = ([
        "__init__.py",
        "//larq_compute_engine/mlir:__init__.py",
        "//larq_compute_engine/tflite:__init__.py",
        "//larq_compute_engine/mlir:python/__init__.py",
    ]),
    deps = [
        "//larq_compute_engine/mlir:converter",
        "//larq_compute_engine/tflite/python:interpreter",
    ],
)
