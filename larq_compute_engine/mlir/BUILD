load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

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
    tblgen = "@llvm-project//mlir:mlir-tblgen",
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
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        # If we remove the dependency on lite:transforms/prepare_patterns.td  we can probably reduce the number of files
        "@org_tensorflow//tensorflow/compiler/mlir/lite:transforms/prepare_patterns.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_optimize_td_files",
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
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
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
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lce_tfl_passes",
    srcs = ["tf_tfl_passes.cc"],
    hdrs = [
        "tf_tfl_passes.h",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:transforms/decode_constant.h",
    ],
    deps = [
        ":larq_compute_engine_optimize",
        ":larq_compute_engine_prepare",
    ],
)

# This takes a lot of time to run
# TODO: Can we remove some dependencies or force TF to only build the necessary stuff?
tf_cc_binary(
    name = "tf_lce_translate",
    srcs = ["tf_lce_translate.cc"],
    deps = [
        ":lce_tfl_passes",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_tfl_translate_cl_options",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "@org_tensorflow//tensorflow/stream_executor/lib",
    ],
)
