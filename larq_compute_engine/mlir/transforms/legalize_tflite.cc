#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

struct LegalizeLCE
    : public PassWrapper<LegalizeLCE, OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final { return "tfl-legalize-lce"; }
  llvm::StringRef getDescription() const final {
    return "Legalize LCE ops in TensorFlow Lite dialect";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }
  void runOnOperation() override;
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
    auto attr = ConstBytesAttr::get(op->getContext(), options_bytes);

    rewriter.replaceOpWithNewOp<CustomOp>(
        op, op->getResultTypes(), op->getOperands(),
        "Lce" + std::string(LarqOp::getOperationName().drop_front(3)), attr);
    return success();
  }
};

void LegalizeLCE::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto* ctx = &getContext();
  auto func = getOperation();

  patterns.add<
      LegalizeToCustomOp<lq::QuantizeOp>, LegalizeToCustomOp<lq::DequantizeOp>,
      LegalizeToCustomOp<lq::Bconv2dOp>, LegalizeToCustomOp<lq::BMaxPool2dOp>>(
      ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the LegalizeLCE pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateLegalizeLCEPass() {
  return std::make_unique<LegalizeLCE>();
}

static PassRegistration<LegalizeLCE> pass;

}  // namespace TFL
}  // namespace mlir
