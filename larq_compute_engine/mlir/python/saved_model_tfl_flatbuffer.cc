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
#include "larq_compute_engine/mlir/tf_tfl_passes.h"
#include "larq_compute_engine/mlir/tf_to_tfl_flatbuffer.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status GetNumInputs(mlir::OwningModuleRef* module, int* num_inputs) {
  *num_inputs = 0;
  mlir::FuncOp entry_function = nullptr;
  for (auto func : module->get().getOps<mlir::FuncOp>()) {
    if (auto tf_attrs =
            func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function")) {
      // TODO(jaesung): There could be multiple entry functions. Let's handle
      // such cases if there are any needs for that.
      if (entry_function != nullptr) {
        return errors::InvalidArgument(
            "There should be only one tf.entry_function");
      }
      entry_function = func;
    }
  }
  if (entry_function == nullptr) {
    return errors::InvalidArgument("no tf.entry_function found");
  }

  // Get the list of input Op names from the function attribute.
  mlir::DictionaryAttr tf_attrs =
      entry_function->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  llvm::SmallVector<llvm::StringRef, 4> function_input_names;
  auto input_attr = tf_attrs.get("inputs");
  if (!input_attr) {
    return errors::InvalidArgument("no inputs attribute found");
  }
  auto input_names = input_attr.cast<mlir::StringAttr>().getValue();
  input_names.split(function_input_names, ",");
  *num_inputs = function_input_names.size();
  return Status::OK();
}

pybind11::bytes ConvertSavedModelToTFLiteFlatBuffer(
    const std::string& saved_model_dir,
    const std::vector<std::string>& saved_model_tags,
    const std::vector<std::string>& exported_names,
    const int saved_model_version, const std::string& target_str,
    const pybind11::object& default_ranges) {
  mlir::MLIRContext context;
  Status status;

  LCETarget target;
  if (target_str == "arm") {
    target = LCETarget::ARM;
  } else if (target_str == "xcore") {
    target = LCETarget::XCORE;
  } else {
    throw std::runtime_error("Invalid target.");
  }

  if (exported_names.size() != 1) {
    throw std::runtime_error("Only a single exported name is supported.");
  }

  tensorflow::GraphImportConfig specs;
  specs.upgrade_legacy = true;

  absl::Span<const std::string> custom_opdefs;

  // Register all custom ops, including user-specified custom ops.
  const toco::TocoFlags toco_flags;
  status = internal::RegisterAllCustomOps(toco_flags);
  if (!status.ok()) {
    throw std::runtime_error(status.error_message());
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
  status = GetNumInputs(&module.ValueOrDie(), &num_inputs);
  if (!status.ok()) {
    throw std::runtime_error(status.error_message());
  }

  mlir::TFL::QuantizationSpecs quant_specs;
  // Normally we'd only set `inference_type` to QINT8 when there are fake_quant
  // nodes in the graph. However this did not work reliably, and even for float
  // models it is fine to set the inference type to QINT8, so we do that by
  // default.
  quant_specs.inference_type = tensorflow::DT_QINT8;
  for (int i = 0; i < num_inputs; ++i) {
    // Input inference type is DT_FLOAT, so set the default input ranges
    // to llvm::None.
    quant_specs.input_ranges.push_back({llvm::None, llvm::None});
  }
  if (!default_ranges.is_none()) {
    // When there are no Quantize nodes in the graph then in the PrepareQuantize
    // pass the variables `eager_quantize` and subsequently `infer_tensor_range`
    // are set to false:
    // https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc#L360-L366
    // This means that the PrepareQuantize pass does *not* infer the int8
    // range of weight tensors. The DefaultQuantParamsPass will then set the
    // quantization stats of those weight tensors to this per-tensor default
    // range instead of proper per-channel ranges.
    // The tflite/tfmicro kernels can handle per-tensor weight quantization, but
    // for some private passes we desire per-channel quantization.
    // To make `infer_tensor_range` become true we simply set
    // `post_training_quantization` to true here.
    // Alternatively to this solution, we could set
    // `quant_specs.target_func = "serving_default";`
    // and set the `input_ranges` to some fixed values. In that case, the
    // PrepareQuantize pass would first insert Quantization ops at the input
    // here:
    // https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc#L172
    // https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc#L202
    quant_specs.post_training_quantization = true;

    quant_specs.default_ranges =
        default_ranges.cast<std::pair<double, double>>();
  }

  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
  tensorflow::SetCrashReproducer(pm);

  tensorflow::AddTFToLCETFLConversionPasses(quant_specs, &pm, target);

  // Convert back to outlined while format for export back to flatbuffer.
  pm.addPass(mlir::TFL::CreateWhileOutlinePass());
  pm.addPass(mlir::TFL::CreateRuntimeVerifyPass());

  std::string result;
  status = ConvertTFExecutorToFlatbuffer(
      module->get(), /*export_to_mlir=*/false, &result, &pm);

  if (!status.ok()) {
    throw std::runtime_error("Could not translate to flatbuffer.");
  }

  return pybind11::bytes(result);
}

}  // namespace tensorflow
