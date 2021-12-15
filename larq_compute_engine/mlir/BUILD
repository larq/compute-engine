load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "lce_ops_td_file",
    srcs = [
        "ir/lce_ops.td",
    ],
    deps = [
        "@llvm-project//mlir:SideEffectTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
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
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
    ],
)

gentbl(
    name = "op_removal_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_op_removal.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/op_removal_patterns.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "prepare_lce_target_arm_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare_target_arm.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_target_arm.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
        "transforms/op_removal_patterns.td",
        "transforms/prepare_patterns_common.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "prepare_lce_target_other_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare_target_other.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_common.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
        "transforms/op_removal_patterns.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "optimize_lce_target_arm_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize_target_arm.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_target_arm.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
        "transforms/optimize_patterns_common.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "optimize_lce_target_other_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize_target_other.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_common.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
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
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
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
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":lce_ops_td_file",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

cc_library(
    name = "larq_compute_engine_bitpack",
    srcs = [
        "transforms/bitpack.cc",
    ],
    hdrs = [
        "transforms/bitpack.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
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
        ":larq_compute_engine_bitpack",
        "//larq_compute_engine/core/bitpacking:bitpack",
        "@flatbuffers",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffects",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
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
        "transforms/generated_prepare_target_arm.inc",
        "transforms/generated_prepare_target_other.inc",
        "transforms/prepare_tf.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "//larq_compute_engine/core:types",
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
        "transforms/generated_optimize_target_arm.inc",
        "transforms/generated_optimize_target_other.inc",
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
        ":larq_compute_engine_bitpack",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
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
    name = "set_batch_size",
    srcs = [
        "transforms/set_batch_size.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        "@llvm-project//mlir:StandardOps",
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
        ":set_batch_size",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:fake_quant_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:decode_constant_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
    ],
)

cc_library(
    name = "tf_to_tfl_flatbuffer",
    srcs = ["tf_to_tfl_flatbuffer.cc"],
    hdrs = [
        "tf_to_tfl_flatbuffer.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/stream_executor/lib",
    ],
)

tf_cc_binary(
    name = "lce-tf-opt",
    srcs = ["lce_mlir_opt.cc"],
    deps = [
        ":lce_tfl_passes",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

pybind_extension(
    name = "_tf_tfl_flatbuffer",
    srcs = [
        "python/graphdef_tfl_flatbuffer.cc",
        "python/pybind_export.cc",
        "python/saved_model_tfl_flatbuffer.cc",
    ],
    module_name = "tf_tfl_flatbuffer",
    deps = [
        ":lce_tfl_passes",
        ":tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/python:tf_tfl_flatbuffer_helpers",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "@org_tensorflow//tensorflow/core:ops",
        "@pybind11",
    ],
)

genrule(
    name = "tflite_schema_py",
    srcs = ["@org_tensorflow//tensorflow/lite/python:schema_py"],
    outs = ["python/tflite_schema.py"],
    cmd = "cp $< $@",
)

py_library(
    name = "converter",
    srcs = [
        "python/converter.py",
        "python/util.py",
        ":tflite_schema_py",
    ],
    deps = [
        ":_tf_tfl_flatbuffer",
    ],
)

exports_files([
    "python/converter.py",
    "python/util.py",
    "__init__.py",
    "python/__init__.py",
])
