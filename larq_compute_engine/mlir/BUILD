load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "lce_ops_td_file",
    srcs = ["ir/lce_ops.td"],
    includes = ["/external/org_tensorflow"],
    deps = [
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
)

td_library(
    name = "fuse_padding_file",
    srcs = ["transforms/fuse_padding.td"],
    includes = ["/external/org_tensorflow"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_inc_gen",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

td_library(
    name = "op_removal_patterns_file",
    srcs = ["transforms/op_removal_patterns.td"],
    includes = ["/external/org_tensorflow"],
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

td_library(
    name = "optimize_patterns_common_file",
    srcs = ["transforms/optimize_patterns_common.td"],
    includes = ["/external/org_tensorflow"],
    deps = [
        ":lce_ops_td_file",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

td_library(
    name = "prepare_patterns_common_file",
    srcs = ["transforms/prepare_patterns_common.td"],
    includes = ["/external/org_tensorflow"],
    deps = [
        ":lce_ops_td_file",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "lce_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/lce_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/lce_ops.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect",
                "lq",
            ],
            "ir/lce_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect",
                "lq",
            ],
            "ir/lce_dialect.cc.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "g3doc/lce_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/lce_ops.td",
    deps = [":lce_ops_td_file"],
)

gentbl_cc_library(
    name = "op_removal_lce_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_op_removal.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/op_removal_patterns.td",
    deps = [":op_removal_patterns_file"],
)

gentbl_cc_library(
    name = "prepare_lce_target_arm_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_prepare_target_arm.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_target_arm.td",
    deps = [":prepare_patterns_common_file"],
)

gentbl_cc_library(
    name = "prepare_lce_target_other_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_prepare_target_other.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_common.td",
    deps = [":prepare_patterns_common_file"],
)

gentbl_cc_library(
    name = "optimize_lce_target_arm_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_optimize_target_arm.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_target_arm.td",
    deps = [":optimize_patterns_common_file"],
)

gentbl_cc_library(
    name = "optimize_lce_target_other_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_optimize_target_other.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_common.td",
    deps = [":optimize_patterns_common_file"],
)

gentbl_cc_library(
    name = "bitpack_activations_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_bitpack_activations.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/bitpack_activations_patterns.td",
    deps = [
        ":lce_ops_td_file",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "bitpack_weights_lce_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_bitpack_weights.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/bitpack_weights_patterns.td",
    deps = [
        ":lce_ops_td_file",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "quantize_lce_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    deps = [
        ":lce_ops_td_file",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "fuse_padding_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_fuse_padding.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/fuse_padding.td",
    deps = [
        ":fuse_padding_file",
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
        "ir/lce_dialect.cc.inc",
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
        "@llvm-project//mlir:SideEffectInterfaces",
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
        "@llvm-project//mlir:FuncDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "larq_compute_engine_prepare",
    srcs = [
        "transforms/common.h",
        "transforms/generated_prepare_target_arm.inc",
        "transforms/generated_prepare_target_other.inc",
        "transforms/prepare_tf.cc",
    ],
    hdrs = [
        "transforms/padding.h",
        "transforms/passes.h",
    ],
    deps = [
        ":larq_compute_engine",
        "//larq_compute_engine/core:types",
        "@llvm-project//mlir:FuncDialect",
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
        "transforms/common.h",
        "transforms/generated_bitpack_activations.inc",
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
    name = "larq_compute_engine_translate_tflite",
    srcs = [
        "transforms/translate_tflite.cc",
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
    name = "detection_postprocess_transform",
    srcs = [
        "transforms/detection_postprocess.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        "//larq_compute_engine/mlir:larq_compute_engine",
        "@llvm-project//mlir:FuncDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fuse_padding",
    srcs = [
        "transforms/fuse_padding.cc",
        "transforms/generated_fuse_padding.inc",
    ],
    hdrs = [
        "transforms/padding.h",
        "transforms/passes.h",
    ],
    deps = [
        "//larq_compute_engine/core:types",
        "@llvm-project//mlir:IR",
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
        "@llvm-project//mlir:FuncDialect",
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
        ":detection_postprocess_transform",
        ":fuse_padding",
        ":larq_compute_engine_bitpack_weights",
        ":larq_compute_engine_legalize_tflite",
        ":larq_compute_engine_op_removal",
        ":larq_compute_engine_optimize",
        ":larq_compute_engine_prepare",
        ":larq_compute_engine_quantize",
        ":larq_compute_engine_translate_tflite",
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
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
    ],
)

cc_library(
    name = "tf_to_tfl_flatbuffer",
    srcs = ["tf_to_tfl_flatbuffer.cc"],
    hdrs = [
        "tf_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":lce_tfl_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/metrics:error_collector",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_freeze_variables",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "python_common",
    srcs = ["python/common.cc"],
    hdrs = ["python/common.h"],
    deps = [
        ":tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/core:ops",
        "@pybind11",
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
        ":python_common",
        ":tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/python:tf_tfl_flatbuffer_helpers",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
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
