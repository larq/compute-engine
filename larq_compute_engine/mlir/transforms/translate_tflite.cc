#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

struct TranslateToLCE : public PassWrapper<TranslateToLCE, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
    registry.insert<mlir::lq::LarqDialect>();
  }
  void runOnFunction() override;
};

struct TranslateToLCEPattern : public OpRewritePattern<TFL::CustomOp> {
  using OpRewritePattern<TFL::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::CustomOp custom_op,
                                PatternRewriter& rewriter) const override {
    auto stringData = custom_op.custom_option().getValue();

    // Replace CustomOp with relevant LarqOp
    if (custom_op.custom_code() == "LceQuantize") {
      rewriter.replaceOpWithNewOp<lq::QuantizeOp>(
          custom_op, custom_op->getResultTypes(), custom_op->getOperands());
    } else if (custom_op.custom_code() == "LceDequantize") {
      rewriter.replaceOpWithNewOp<lq::DequantizeOp>(
          custom_op, custom_op->getResultTypes(), custom_op->getOperands());
    } else if (custom_op.custom_code() == "LceBMaxPool2d") {
      auto map =
          flexbuffers::GetRoot((uint8_t*)stringData.data(), stringData.size())
              .AsMap();
      rewriter.replaceOpWithNewOp<lq::BMaxPool2dOp>(
          custom_op, custom_op->getResultTypes(), custom_op->getOperands()[0],
          stringifyPadding(static_cast<Padding>(map["padding"].AsInt32())),
          map["stride_width"].AsInt32(), map["stride_height"].AsInt32(),
          map["filter_width"].AsInt32(), map["filter_height"].AsInt32());
    } else if (custom_op.custom_code() == "LceBconv2d") {
      auto map =
          flexbuffers::GetRoot((uint8_t*)stringData.data(), stringData.size())
              .AsMap();
      rewriter.replaceOpWithNewOp<lq::Bconv2dOp>(
          custom_op, custom_op->getResultTypes(), custom_op->getOperands()[0],
          custom_op->getOperands()[1], custom_op->getOperands()[2],
          custom_op->getOperands()[3], custom_op->getOperands()[4],
          map["channels_in"].AsInt32(), map["dilation_height_factor"].AsInt32(),
          map["dilation_width_factor"].AsInt32(),
          stringifyActivationFunctionType(static_cast<ActivationFunctionType>(
              map["fused_activation_function"].AsInt32())),
          map["pad_values"].AsInt32(),
          stringifyPadding(static_cast<Padding>(map["padding"].AsInt32())),
          map["stride_height"].AsInt32(), map["stride_width"].AsInt32());
    }

    return success();
  }
};

void TranslateToLCE::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto* ctx = &getContext();
  auto func = getFunction();
  patterns.insert<TranslateToLCEPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TranslateToLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateTranslateToLCEPass() {
  return std::make_unique<TranslateToLCE>();
}

static PassRegistration<TranslateToLCE> pass(
    "lce-translate-tfl", "Translate TFL custom ops to LCE ops");

}  // namespace TFL
}  // namespace mlir
