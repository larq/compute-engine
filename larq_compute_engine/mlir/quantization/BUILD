load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mlir_quantize_model",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
    ],
    deps = [
        "//larq_compute_engine/mlir:larq_compute_engine_sanitize",
        "@flatbuffers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:common",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
    ],
)

pybind_extension(
    name = "_calibration_wrapper",
    srcs = ["calibration_wrapper.cc"],
    module_name = "calibration_wrapper",
    deps = [
        ":mlir_quantize_model",
        ":tflite_quantize_model",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/python/interpreter_wrapper:numpy",
        "@org_tensorflow//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "@org_tensorflow//tensorflow/lite/tools/optimize/calibration:calibration_reader",
        "@org_tensorflow//tensorflow/lite/tools/optimize/calibration:calibrator_lib",
        "@pybind11",
    ],
)

#
# We require tflite::optimize::QuantizeModel
# defined in
#     lite/tools/optimize/quantize_model.cc
# Out of all the files in this folder we only need
# to modify
#     lite/tools/optimize/operator_property.cc
# So we refer to @org_tensorflow for everything except
# all build rules that use ":operator_property", which we
# replace by our own
# So we copy the build rules of everything but replace
# only that file
#

cc_library(
    name = "operator_property",
    srcs = ["operator_property.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/optimize:operator_property.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

# Although we do not modify model_utils or quantization_utils, they do depend on operator_property so we have to re-define their build rules but we can refer to the original source files

cc_library(
    name = "tflite_quantize_model",
    srcs = ["@org_tensorflow//tensorflow/lite/tools/optimize:quantize_model.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/optimize:quantize_model.h"],
    deps = [
        ":model_utils",
        ":operator_property",
        ":quantization_utils",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:util",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["@org_tensorflow//tensorflow/lite/tools/optimize:model_utils.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/optimize:model_utils.h"],
    deps = [
        ":operator_property",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["@org_tensorflow//tensorflow/lite/tools/optimize:quantization_utils.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/optimize:quantization_utils.h"],
    deps = [
        ":model_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:minimal_logging",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/kernels/internal:quantization_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:round",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//third_party/eigen3",
    ],
)
