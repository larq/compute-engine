#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

struct LegalizeLCE : public PassWrapper<LegalizeLCE, FunctionPass> {
  void runOnFunction() override;
};

template <typename LarqOp>
struct LegalizeToCustomOp : public OpRewritePattern<LarqOp> {
  using OpRewritePattern<LarqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LarqOp larq_op,
                                PatternRewriter& rewriter) const override {
    std::vector<uint8_t> options = larq_op.buildCustomOptions();
    Operation* op = larq_op.getOperation();
    ShapedType type = RankedTensorType::get(
        {static_cast<int64_t>(options.size())}, rewriter.getIntegerType(8));

    std::string options_bytes(options.begin(), options.end());
    auto attr = OpaqueElementsAttr::get(op->getDialect(), type, options_bytes);

    rewriter.replaceOpWithNewOp<CustomOp>(
        op, op->getResultTypes(), op->getOperands(),
        "Lce" + std::string(LarqOp::getOperationName().drop_front(3)), attr);
    return success();
  }
};

void LegalizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  patterns.insert<LegalizeToCustomOp<TF::BsignOp>,
                  LegalizeToCustomOp<TF::Bconv2dOp>,
                  LegalizeToCustomOp<TF::BMaxPool2dOp>>(ctx);

  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the LegalizeLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeLCEPass() {
  return std::make_unique<LegalizeLCE>();
}

static PassRegistration<LegalizeLCE> pass(
    "tfl-legalize-lce", "Legalize LCE ops in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
