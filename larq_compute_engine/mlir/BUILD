load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "lce_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/lce_ops.h.inc"),
        ("-gen-op-defs", "ir/lce_ops.cc.inc"),
        ("-gen-dialect-decls -dialect=lq", "ir/lce_dialect.h.inc"),
        ("-gen-dialect-doc", "g3doc/lce_ops.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/lce_ops.td",
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "op_removal_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_op_removal.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/op_removal_patterns.td",
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "prepare_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/op_removal_patterns.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "optimize_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "bitpack_weights_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_bitpack_weights.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/bitpack_weights_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/bitpack_weights_patterns.td",
        "transforms/op_removal_patterns.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "quantize_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_quantize.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

cc_library(
    name = "larq_compute_engine",
    srcs = [
        "ir/lce_dialect.h.inc",
        "ir/lce_ops.cc",
        "ir/lce_ops.cc.inc",
        "ir/lce_ops.h.inc",
    ],
    hdrs = [
        "ir/lce_ops.h",
        "transforms/passes.h",
    ],
    deps = [
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@flatbuffers",
        "@llvm-project//mlir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# Library with tensorflow Lite dialect static initialization.
cc_library(
    name = "larq_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":larq_compute_engine",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_op_removal",
    srcs = [
        "transforms/generated_op_removal.inc",
        "transforms/op_removal.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        "@llvm-project//mlir:StandardOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_prepare",
    srcs = [
        "transforms/generated_prepare.inc",
        "transforms/prepare_tf.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "@llvm-project//mlir:StandardOps",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:validators",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_optimize",
    srcs = [
        "transforms/generated_optimize.inc",
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_bitpack_weights",
    srcs = [
        "transforms/bitpack_weights.cc",
        "transforms/generated_bitpack_weights.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "//larq_compute_engine/core:types",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_legalize_tflite",
    srcs = [
        "transforms/legalize_tflite.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_quantize",
    srcs = [
        "transforms/generated_quantize.inc",
        "transforms/quantize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lce_tfl_passes",
    srcs = ["tf_tfl_passes.cc"],
    hdrs = [
        "tf_tfl_passes.h",
    ],
    deps = [
        ":larq_compute_engine_bitpack_weights",
        ":larq_compute_engine_legalize_tflite",
        ":larq_compute_engine_op_removal",
        ":larq_compute_engine_optimize",
        ":larq_compute_engine_prepare",
        ":larq_compute_engine_quantize",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:decode_constant_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
    ],
)

cc_binary(
    name = "lce-tf-opt",
    deps = [
        ":larq_dialect_registration",
        ":lce_tfl_passes",
        "@llvm-project//mlir:MlirOptMain",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
    ],
)

pybind_extension(
    name = "_graphdef_tfl_flatbuffer",
    srcs = ["python/graphdef_tfl_flatbuffer.cc"],
    module_name = "graphdef_tfl_flatbuffer",
    deps = [
        ":larq_dialect_registration",
        ":lce_tfl_passes",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "@org_tensorflow//tensorflow/core:ops",
        "@pybind11",
    ],
)

py_library(
    name = "converter",
    srcs = [
        "python/converter.py",
        "python/schema_py_generated.py",
        "python/util.py",
    ],
    deps = [
        ":_graphdef_tfl_flatbuffer",
    ],
)

exports_files([
    "python/converter.py",
    "python/util.py",
    "python/schema_py_generated.py",
    "__init__.py",
    "python/__init__.py",
])
