# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2020 Larq Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Forked from tensorflow/lite/tools/evaluation/stages/BUILD.
#
# Modified to strip out most build targets, and to inject the correct
# dependencies for those that remain.

load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_inference_stage",
    srcs = ["tflite_inference_stage.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/evaluation/stages:tflite_inference_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/core/util:stats_calculator_portable",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/profiling:time",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:evaluation_stage",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

cc_library(
    name = "image_classification_stage",
    srcs = ["image_classification_stage.cc"],
    hdrs = ["@org_tensorflow//tensorflow/lite/tools/evaluation/stages:image_classification_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":tflite_inference_stage",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:evaluation_stage",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/stages:image_preprocessing_stage",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/stages:topk_accuracy_eval_stage",
    ],
)
