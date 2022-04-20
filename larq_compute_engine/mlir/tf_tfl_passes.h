#ifndef LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_

#include <functional>

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"

namespace tensorflow {

void AddPreVariableFreezingTFToLCETFLConversionPasses(
    mlir::OpPassManager* pass_manager);

void AddPostVariableFreezingTFToLCETFLConversionPasses(
    llvm::StringRef saved_model_dir,
    const mlir::TFL::QuantizationSpecs& quant_specs,
    mlir::OpPassManager* pass_manager, const LCETarget target = LCETarget::ARM);

}  // namespace tensorflow

#endif  // LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_
