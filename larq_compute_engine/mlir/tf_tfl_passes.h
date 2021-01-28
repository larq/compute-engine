#ifndef LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_

#include <functional>

#include "larq_compute_engine/mlir/transforms/passes.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"

namespace tensorflow {

// Add the TF to TFLite passes into a pass_manager.
void AddTFToLCETFLConversionPasses(
    const mlir::TFL::QuantizationSpecs& quant_specs,
    mlir::OpPassManager* pass_manager, const LCETarget target = LCETarget::ARM,
    const bool experimental_enable_bitpacked_activations = false);

}  // namespace tensorflow

#endif  // LARQ_COMPUTE_ENGINE_MLIR_TF_TFL_PASSES_H_
