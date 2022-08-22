#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/bitpack.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

struct BitpackWeightsLCE
    : public PassWrapper<BitpackWeightsLCE, OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final {
    return "tfl-lce-bitpack-weights";
  }
  llvm::StringRef getDescription() const final {
    return "Bitpack binary weights";
  }
  void runOnOperation() override;
};

bool IsConv2DFilter(Attribute filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;
  auto filter_type = filter.getType().cast<ShapedType>();
  return filter_type.getElementType().isF32() &&
         filter_type.getShape().size() == 4;
}

#include "larq_compute_engine/mlir/transforms/generated_bitpack_weights.inc"

void BitpackWeightsLCE::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  TFL::populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow dialect BitpackWeights pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
CreateBitpackWeightsLCEPass() {
  return std::make_unique<BitpackWeightsLCE>();
}

static PassRegistration<BitpackWeightsLCE> pass;

}  // namespace TFL
}  // namespace mlir
