#include "larq_compute_engine/mlir/transforms/padding.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Version used when matching a TFLite op that has `stride_height` and
// `stride_width` as separate attributes. Due to a TableGen limitation we can't
// pass them both into a single function call.
bool IsSamePaddingPartial(Attribute paddings_attr, Value input, Value output,
                          Attribute strides_attr, uint64_t dimension) {
  auto paddings = GetValidPadAttr(paddings_attr);
  if (!paddings) return false;
  auto input_shape = GetShape4D(input);
  if (input_shape.empty()) return false;
  auto output_shape = GetShape4D(output);
  if (output_shape.empty()) return false;

  if (!strides_attr.isa<IntegerAttr>()) return false;
  const int stride = strides_attr.cast<IntegerAttr>().getInt();

  // Check that there is no padding in the batch and channel dimensions
  return IsNoPadding(paddings, 0) &&
         IsSamePadding1D(paddings, dimension, input_shape[dimension],
                         output_shape[dimension], stride) &&
         IsNoPadding(paddings, 3);
}

#include "larq_compute_engine/mlir/transforms/generated_fuse_padding.inc"

// Prepare LCE operations in functions for subsequent legalization.
struct FusePadding : public PassWrapper<FusePadding, FunctionPass> {
  FusePadding() = default;
  FusePadding(const FusePadding& pass) {}

  void runOnFunction() override {
    auto* ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    auto func = getFunction();
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<::mlir::TFL::TensorFlowLiteDialect>();
  }
};

}  // namespace

// Creates an instance of the TensorFlow dialect FusePadding pass.
std::unique_ptr<OperationPass<FuncOp>> CreateFusePaddingPass() {
  return std::make_unique<FusePadding>();
}

static PassRegistration<FusePadding> pass(
    "tfl-fuse-padding", "Fuse padding ops into (Depthwise)Convs.");

}  // namespace TFL
}  // namespace mlir
