#ifndef LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_

#include <optional>
#include <unordered_set>

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/core/public/session.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// This is a fork of ConvertTFExecutorToTFLOrFlatbuffer to enable custom
// OpOrArgLocNameMapper
// https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h#L60-L78
Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, const LCETarget target,
    mlir::quant::QuantizationSpecs quant_specs,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir,
    std::optional<tensorflow::Session*> session, std::string* result);
}  // namespace tensorflow

#endif  // LARQ_COMPUTE_ENGINE_MLIR_TF_TO_TFL_FLATBUFFER_H_
