#ifndef LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_
#define LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_

#include "mlir/Pass/Pass.h"

enum LCETarget { ARM = 0, XCORE = 1 };

namespace mlir {
namespace TFL {

// Creates an instance of the TensorFlow dialect OpRemoval pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOpRemovalPass();

// Creates an instance of the TensorFlow dialect PrepareLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareLCEPass(LCETarget target);

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeLCEPass(LCETarget target);

// Creates an instance of the TensorFlow dialect BitpackWeightsLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateBitpackWeightsLCEPass();

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLCEQuantizePass();

// Creates an instance of LegalizeLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeLCEPass();

// Creates an instance of TranslateToLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateTranslateToLCEPass();

}  // namespace TFL

// Creates an instance of the TensorFlow dialect SetBatchSize pass
std::unique_ptr<OperationPass<FuncOp>> CreateSetBatchSizePass();

}  // namespace mlir

#endif  // LARQ_COMPUTE_ENGINE_MLIR_PASSES_H_
