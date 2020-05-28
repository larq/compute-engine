#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

std::vector<uint8_t> buildCustomOption(TF::BsignOp op) { return {}; }

std::vector<uint8_t> buildCustomOption(TF::Bconv2dOp op) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("channels_in", op.channels_in().getSExtValue());
    fbb.Int("dilation_height_factor",
            op.dilation_height_factor().getSExtValue());
    fbb.Int("dilation_width_factor", op.dilation_width_factor().getSExtValue());
    fbb.String("fused_activation_function",
               std::string(op.fused_activation_function()));
    fbb.Int("pad_values", op.pad_values().getSExtValue());
    fbb.String("padding", std::string(op.padding()));
    fbb.Int("stride_height", op.stride_height().getSExtValue());
    fbb.Int("stride_width", op.stride_width().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> buildCustomOption(TF::BMaxPool2dOp op) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.String("padding", std::string(op.padding()));
    fbb.Int("stride_width", op.stride_width().getSExtValue());
    fbb.Int("stride_height", op.stride_height().getSExtValue());
    fbb.Int("filter_width", op.filter_width().getSExtValue());
    fbb.Int("filter_height", op.filter_height().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

struct LegalizeLCE : public PassWrapper<LegalizeLCE, FunctionPass> {
  void runOnFunction() override;
};

template <typename LarqOp>
struct LegalizeToCustomOp : public OpRewritePattern<LarqOp> {
  using OpRewritePattern<LarqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LarqOp larq_op,
                                PatternRewriter& rewriter) const override {
    std::vector<uint8_t> options = buildCustomOption(larq_op);
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
