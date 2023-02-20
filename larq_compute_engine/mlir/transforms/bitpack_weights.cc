#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/bitpack.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

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

bool IsConv2DFilter(TypedAttr filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;
  auto filter_type = filter.getType().cast<ShapedType>();
  return filter_type.getElementType().isF32() &&
         filter_type.getShape().size() == 4;
}

namespace bitpackweights {
#include "larq_compute_engine/mlir/transforms/generated_bitpack_weights.inc"
}  // namespace bitpackweights

void BitpackWeightsLCE::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  bitpackweights::populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

// Creates an instance of the TensorFlow dialect BitpackWeights pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
CreateBitpackWeightsLCEPass() {
  return std::make_unique<BitpackWeightsLCE>();
}

static PassRegistration<BitpackWeightsLCE> pass;

}  // namespace TFL
}  // namespace mlir
