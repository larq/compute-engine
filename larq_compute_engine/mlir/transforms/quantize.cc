// This transformation pass applies hybrid quantization on TFLite dialect.

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

// Integer quantization rewrite pattern for TFLite.
// We allow hybrid operands so post_activation_multiplier and
// post_activation_bias can be kept in float32 for now
struct TFLHybridQuantization
    : public quant::QuantizationPattern<TFLHybridQuantization, QuantizeOp,
                                        DequantizeOp, NumericVerifyOp> {
  explicit TFLHybridQuantization(MLIRContext* ctx, bool verify_numeric,
                                 float tolerance, bool verify_single_layer)
      : BaseType(ctx, verify_numeric, tolerance, verify_single_layer) {}
  static bool AllowHybridOperand() { return true; }
  static bool AllowHybridResult() { return false; }
};

// Applies quantization on the model in TFL dialect.
struct HybridQuantizePass
    : public PassWrapper<HybridQuantizePass, FunctionPass> {
  void runOnFunction() override;
};

#include "larq_compute_engine/mlir/transforms/generated_quantize.inc"

void HybridQuantizePass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<TFLHybridQuantization>(ctx, /*enable_numeric_verify=*/false,
                                         /*error_tolerance=*/5.0,
                                         /*enable_single_layer_verify=*/true);
  applyPatternsAndFoldGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateHybridQuantizePass() {
  return std::make_unique<HybridQuantizePass>();
}

static PassRegistration<HybridQuantizePass> pass(
    "tfl-hybrid-quantize",
    "Apply hybrid quantization on models in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
