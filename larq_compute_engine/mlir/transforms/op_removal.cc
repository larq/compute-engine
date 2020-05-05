#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Op removal of pass through ops to make following patterns easier and enable
// early constant folding
struct OpRemoval : public PassWrapper<OpRemoval, FunctionPass> {
  void runOnFunction() override;
};

#include "larq_compute_engine/mlir/transforms/generated_op_removal.inc"

void OpRemoval::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OpRemoval pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOpRemovalPass() {
  return std::make_unique<OpRemoval>();
}

static PassRegistration<OpRemoval> pass("lce-op-removal-tf",
                                        "Remove pass through TensorFlow ops");

}  // namespace TFL
}  // namespace mlir
