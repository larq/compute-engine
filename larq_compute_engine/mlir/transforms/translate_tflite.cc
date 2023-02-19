#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

static llvm::StringRef ConvertActivationAttr(
    tflite::ActivationFunctionType af_type) {
  if (af_type == tflite::ActivationFunctionType_NONE) return "NONE";
  if (af_type == tflite::ActivationFunctionType_RELU) return "RELU";
  if (af_type == tflite::ActivationFunctionType_RELU_N1_TO_1)
    return "RELU_N1_TO_1";
  if (af_type == tflite::ActivationFunctionType_RELU6) return "RELU6";
}

static llvm::StringRef ConvertPaddingAttr(tflite::Padding padding_type) {
  if (padding_type == tflite::Padding_SAME) return "SAME";
  if (padding_type == tflite::Padding_VALID) return "VALID";
}

namespace mlir {
namespace TFL {

struct TranslateToLCE
    : public PassWrapper<TranslateToLCE, OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final { return "lce-translate-tfl"; }
  llvm::StringRef getDescription() const final {
    return "Translate TFL custom ops to LCE ops";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::lq::LarqDialect>();
  }
  void runOnOperation() override;
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
          custom_op, custom_op->getResultTypes(), custom_op->getOperand(0),
          ConvertPaddingAttr(
              static_cast<tflite::Padding>(map["padding"].AsInt32())),
          map["stride_width"].AsInt32(), map["stride_height"].AsInt32(),
          map["filter_width"].AsInt32(), map["filter_height"].AsInt32());
    } else if (custom_op.custom_code() == "LceBconv2d") {
      auto map =
          flexbuffers::GetRoot((uint8_t*)stringData.data(), stringData.size())
              .AsMap();
      rewriter.replaceOpWithNewOp<lq::Bconv2dOp>(
          custom_op, custom_op->getResultTypes(), custom_op->getOperand(0),
          custom_op->getOperand(1), custom_op->getOperand(2),
          custom_op->getOperand(3), custom_op->getOperand(4),
          map["channels_in"].AsInt32(), map["dilation_height_factor"].AsInt32(),
          map["dilation_width_factor"].AsInt32(),
          ConvertActivationAttr(static_cast<tflite::ActivationFunctionType>(
              map["fused_activation_function"].AsInt32())),
          map["pad_values"].AsInt32(),
          ConvertPaddingAttr(
              static_cast<tflite::Padding>(map["padding"].AsInt32())),
          map["stride_height"].AsInt32(), map["stride_width"].AsInt32());
    }

    return success();
  }
};

void TranslateToLCE::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto* ctx = &getContext();
  auto func = getOperation();
  patterns.add<TranslateToLCEPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

// Creates an instance of the TranslateToLCE pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateTranslateToLCEPass() {
  return std::make_unique<TranslateToLCE>();
}

static PassRegistration<TranslateToLCE> pass;

}  // namespace TFL
}  // namespace mlir
