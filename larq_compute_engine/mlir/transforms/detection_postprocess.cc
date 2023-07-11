#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

struct DetectionPostProcess
    : public PassWrapper<DetectionPostProcess,
                         OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final {
    return "detection-postprocess-int";
  }
  llvm::StringRef getDescription() const final {
    return "Make detection postprocessing op run with int8 input";
  }
  void runOnOperation() override;
};

struct RemoveDequantizeBeforePostProcess : public OpRewritePattern<CustomOp> {
  using OpRewritePattern<CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomOp detection_op,
                                PatternRewriter& rewriter) const override {
    // ----------------- matching part -----------------

    // Match the custom op code to 'TFLite_Detection_PostProcess'
    auto custom_code = detection_op.custom_code().str();
    if (custom_code != "TFLite_Detection_PostProcess") {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "op 'tfl.custom' attribute 'custom_code' failed to satisfy "
                "constraint: constant attribute TFLite_Detection_PostProcess";
      });
    }

    // Check the number of inputs and outputs of the detection op
    auto original_detection_inputs = detection_op.input();
    if (original_detection_inputs.size() != 3) {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "expected 3 inputs for the detection op";
      });
    }
    auto original_detection_outputs = detection_op.output();
    if (original_detection_outputs.size() != 4) {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "expected 4 outputs for the original detection op";
      });
    }

    // Match that dequantization happens just before the detection op
    auto boxes_input_op = original_detection_inputs[0].getDefiningOp();
    auto original_boxes_dequantize_op =
        llvm::dyn_cast_or_null<DequantizeOp>(boxes_input_op);
    if (!original_boxes_dequantize_op) {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "expected dequantization before the op for the 'boxes' input";
      });
    }
    auto scores_input_op = original_detection_inputs[1].getDefiningOp();
    auto original_scores_dequantize_op =
        llvm::dyn_cast_or_null<DequantizeOp>(scores_input_op);
    if (!original_scores_dequantize_op) {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "expected dequantization before the op for the 'scores' input";
      });
    }
    auto anchors_input_op = original_detection_inputs[2].getDefiningOp();
    auto original_anchors_dequantize_op =
        llvm::dyn_cast_or_null<DequantizeOp>(anchors_input_op);
    if (!original_anchors_dequantize_op) {
      return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
        diag << "expected dequantization before the op for the 'anchors' input";
      });
    }

    // Verify the output types of the current detection op:
    // Output type #0: [int32] detection boxes (scaled by 2048)
    // Output type #1: [int32] detection class IDs
    // Output type #2: [float32] detection scores
    // Output type #3: [int32] number of detections
    auto output_data_types = SmallVector<Type, 4>{
        rewriter.getIntegerType(32),
        rewriter.getIntegerType(32),
        rewriter.getF32Type(),
        rewriter.getIntegerType(32),
    };
    for (auto i = 0; i < 4; ++i) {
      auto original_type =
          original_detection_outputs[i].getType().cast<ShapedType>();
      auto original_data_type = original_type.getElementType();
      if (output_data_types[i] != original_data_type) {
        return rewriter.notifyMatchFailure(detection_op, [&](Diagnostic& diag) {
          diag << "unexpected output type of the op";
        });
      }
    }

    // ----------------- re-write part -----------------

    // Get the original inputs (before dequantization)
    auto boxes_input = original_boxes_dequantize_op.input();
    auto scores_input = original_scores_dequantize_op.input();
    auto anchors_input = original_anchors_dequantize_op.input();

    // Set the new detection inputs
    auto new_detection_inputs =
        SmallVector<Value, 3>{boxes_input, scores_input, anchors_input};

    // Set the 4 outputs types [scores, classes, boxes, num_detections]:
    // Output type #0: [int32] detection boxes (scaled by 2048)
    // Output type #1: [int32] detection class IDs
    // Output type #2: [int8 quantized] detection scores
    // Output type #3: [int32] number of detections
    // All as before, except for output #2 (float -> int8 quantized)
    auto scores_type = scores_input.getType()
                           .cast<ShapedType>()
                           .getElementType()
                           .cast<quant::UniformQuantizedType>();
    const auto scores_zp = scores_type.getZeroPoint();
    const auto scores_scale = scores_type.getScale();
    output_data_types[2] = quant::UniformQuantizedType::get(
        true, rewriter.getIntegerType(8), rewriter.getF32Type(), scores_scale,
        scores_zp, -128, 127);

    // Set for all the outputs: data-type (as set above) and shape (as before)
    auto new_op_output_types = SmallVector<Type, 4>{};
    for (auto i = 0; i < 4; ++i) {
      auto value = original_detection_outputs[i];
      auto shape = value.getType().cast<ShapedType>().getShape();
      auto new_output_type = RankedTensorType::get(shape, output_data_types[i]);
      new_op_output_types.push_back(new_output_type);
    }

    // Add a new detection op (with int8 input and int8/int32 output)
    auto new_detection_op = rewriter.create<CustomOp>(
        detection_op->getLoc(), new_op_output_types, new_detection_inputs,
        std::string{"TFLite_Detection_PostProcess"},
        detection_op.custom_option());

    // Add the 4 outputs: boxes, classes, scores, num_detections
    auto new_outputs = SmallVector<Value, 4>{};

    // Output #0: [int32] detection boxes (scaled by 2048)
    new_outputs.push_back(new_detection_op.output()[0]);

    // Output #1: [int32] detection class IDs
    new_outputs.push_back(new_detection_op.output()[1]);

    // Output #2: [int8 quantized] detection scores
    auto new_dequantize_op = rewriter.create<DequantizeOp>(
        detection_op->getLoc(), original_detection_outputs[2].getType(),
        new_detection_op.output()[2]);
    new_outputs.push_back(new_dequantize_op.output());

    // Output #3: [int32] number of detections
    new_outputs.push_back(new_detection_op.output()[3]);

    // Final re-write of the detection op with detection + quantization
    rewriter.replaceOp(detection_op, new_outputs);
    return success();
  };
};

void DetectionPostProcess::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto func = getOperation();

  patterns.add<RemoveDequantizeBeforePostProcess>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

// Creates an instance of the TensorFlow dialect DetectionPostProcess pass.
std::unique_ptr<OperationPass<func::FuncOp>>
QuantizeDetectionPostProcessPass() {
  return std::make_unique<DetectionPostProcess>();
}

static PassRegistration<DetectionPostProcess> pass;

}  // namespace TFL
}  // namespace mlir
