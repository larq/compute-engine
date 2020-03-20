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
        ("-gen-op-doc", "g3doc/lce_ops.md"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/lce_ops.td",
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "prepare_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        # If we remove the dependency on lite:transforms/prepare_patterns.td  we can probably reduce the number of files
        "@org_tensorflow//tensorflow/compiler/mlir/lite:transforms/prepare_patterns.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_optimize_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "optimize_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
    ],
)

cc_library(
    name = "larq_compute_engine",
    srcs = [
        "ir/lce_ops.cc",
        "ir/lce_ops.cc.inc",
        "ir/lce_ops.h.inc",
    ],
    hdrs = [
        "ir/lce_ops.h",
        "transforms/passes.h",
    ],
    deps = [
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
        "transforms/utils.h",
    ],
    deps = [
        ":larq_compute_engine",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
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
        "transforms/utils.h",
    ],
    deps = [
        ":larq_compute_engine",
        "//larq_compute_engine/core:packbits",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

# This is part of @org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes
# which includes most of TensorFlow. We extract this single target to keep
# binary size and build time manageable.
cc_library(
    name = "tf_decode_constant_pass",
    srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:transforms/decode_constant.cc",
    ],
    hdrs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:transforms/decode_constant.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor",
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
        ":larq_compute_engine_optimize",
        ":larq_compute_engine_prepare",
        ":tf_decode_constant_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
    ],
)

cc_binary(
    name = "lce-tf-opt",
    deps = [
        ":lce_tfl_passes",
        "@local_config_mlir//:MlirOptMain",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
    ],
)

pybind_extension(
    name = "_graphdef_tfl_flatbuffer",
    srcs = ["python/graphdef_tfl_flatbuffer.cc"],
    module_name = "graphdef_tfl_flatbuffer",
    deps = [
        ":lce_tfl_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/core:ops",
        "@pybind11",
    ],
)

py_library(
    name = "converter",
    srcs = ["python/converter.py"],
    deps = [
        ":_graphdef_tfl_flatbuffer",
    ],
)

exports_files([
    "python/converter.py",
    "__init__.py",
    "python/__init__.py",
])
