#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public FunctionPass<OptimizeLCE> {
  void runOnFunction() override;
};

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  // Cleanup dead ops manually. LCE ops are not registered to the TF dialect so
  // op->hasNoSideEffect() will return false. Therefor applyPatternsGreedily
  // won't automatically remove the dead nodes. See
  // https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/IR/Operation.h#L457-L462
  patterns.insert<mlir::CleanupDeadOps<TF::LqceBconv2d64Op>>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizeLCEPass() {
  return std::make_unique<OptimizeLCE>();
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
