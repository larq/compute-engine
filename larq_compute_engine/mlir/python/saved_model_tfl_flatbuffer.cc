/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Modifications copyright (C) 2021 Larq Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <exception>
#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "larq_compute_engine/mlir/python/common.h"
#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

pybind11::bytes ConvertSavedModelToTFLiteFlatBuffer(
    const std::string& saved_model_dir,
    const std::vector<std::string>& saved_model_tags,
    const std::vector<std::string>& exported_names,
    const int saved_model_version, const std::string& target_str,
    const pybind11::object& default_ranges) {
  mlir::MLIRContext context;
  Status status;

  auto target = GetLCETarget(target_str);

  if (exported_names.empty()) {
    throw std::runtime_error("Need at least one exported name.");
  }

  tensorflow::GraphImportConfig specs;
  specs.upgrade_legacy = true;

  absl::Span<const std::string> custom_opdefs;

  // Register all custom ops, including user-specified custom ops.
  const toco::TocoFlags toco_flags;
  status = internal::RegisterAllCustomOps(toco_flags);
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }

  // Some weirdness required to convert the vector<string> to an
  // absl::Span<std::string>
  auto exported_names_vector =
      std::vector<std::string>(exported_names.begin(), exported_names.end());
  absl::Span<std::string> exported_names_span(exported_names_vector);

  std::unordered_set<std::string> tags(saved_model_tags.begin(),
                                       saved_model_tags.end());

  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();
  auto module =
      ImportSavedModel(saved_model_dir, saved_model_version, tags,
                       custom_opdefs, exported_names_span, specs,
                       /*enable_variable_lifting=*/true, &context, &bundle);

  if (!module.ok()) {
    throw std::runtime_error("Could not import SavedModel.");
  }

  int num_inputs = 0;
  status = GetNumInputs(&module.value(), &num_inputs);
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }

  return ConvertMLIRModuleToTFLiteFlatBuffer(
      &module.value(), context, target, default_ranges, tags,
      saved_model_dir, bundle ? bundle->GetSession() : nullptr, num_inputs,
      /*should_quantize=*/true,
      /*mark_as_post_training_quant=*/true);
}

}  // namespace tensorflow
