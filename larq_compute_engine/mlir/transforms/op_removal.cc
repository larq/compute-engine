#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Op removal of pass through ops to make following patterns easier and enable
// early constant folding
struct OpRemoval : public PassWrapper<OpRemoval, OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final { return "lce-op-removal-tf"; }
  llvm::StringRef getDescription() const final {
    return "Remove pass through TensorFlow ops";
  }
  void runOnOperation() override;
};

#include "larq_compute_engine/mlir/transforms/generated_op_removal.inc"

void OpRemoval::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  TFL::populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow dialect OpRemoval pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateOpRemovalPass() {
  return std::make_unique<OpRemoval>();
}

static PassRegistration<OpRemoval> pass;

}  // namespace TFL
}  // namespace mlir
