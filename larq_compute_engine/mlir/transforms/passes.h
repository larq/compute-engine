#ifndef LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_
#define LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace TFL {

// Creates an instance of the TensorFlow dialect PrepareLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareLCEPass();

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizeLCEPass(
    bool experimental_enable_bitpacked_activations);

// Creates an instance of SanitizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateSanitizeLCEPass();

}  // namespace TFL
}  // namespace mlir

#endif  // LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_
