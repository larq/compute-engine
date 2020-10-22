// This transformation pass applies quantization on Larq dialect.

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

// Applies quantization on the model in TFL dialect.
struct LCEQuantizePass : public PassWrapper<LCEQuantizePass, FunctionPass> {
  void runOnFunction() override;
};

#include "larq_compute_engine/mlir/transforms/generated_quantize.inc"

void LCEQuantizePass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(ctx, patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLCEQuantizePass() {
  return std::make_unique<LCEQuantizePass>();
}

static PassRegistration<LCEQuantizePass> pass(
    "lce-quantize",
    "Apply hybrid quantization on models in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
