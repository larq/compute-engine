#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Prepare LCE operations in functions for subsequent legalization.
struct PrepareLCE : public FunctionPass<PrepareLCE> {
  void runOnFunction() override;
};

// The bconv will do the following:
// output = fused_add[channel] + fused_multiply[channel] * popcount
// We use this to implement two things:
// - `y1 = n - 2 * popcount`     (the backtransformation to -1,+1 space)
// - `y2 = a + b * y1`           (optional fused batchnorm)
// Together they become
// `y = (a + b*n) + (-2b) * popcount

DenseElementsAttr GetBias(Value filter) {
  auto filter_type = filter.getType().cast<ShapedType>();
  auto filter_shape = filter_type.getShape();
  // Here the weights are still HWIO
  auto dotproduct_size = filter_shape[0] * filter_shape[1] * filter_shape[2];

  RankedTensorType type =
      RankedTensorType::get({filter_shape[3]}, filter_type.getElementType());
  return DenseElementsAttr::get(type, static_cast<float_t>(dotproduct_size));
}

DenseElementsAttr GetMultiplier(Value filter) {
  auto filter_type = filter.getType().cast<ShapedType>();
  auto filter_shape = filter_type.getShape();

  RankedTensorType type =
      RankedTensorType::get({filter_shape[3]}, filter_type.getElementType());
  return DenseElementsAttr::get(type, -2.0f);
}

static bool IsConst(Operation* op) {
  return isa<ConstantOp>(op) || isa<TF::ConstOp>(op) || isa<ConstOp>(op) ||
         isa<QConstOp>(op);
}

bool IsBinaryFilter(Value* filter) {
  auto op = filter->getDefiningOp();
  if (!IsConst(op)) return false;

  auto tensor_attr = op->getAttr("value").cast<DenseElementsAttr>();
  for (auto value : tensor_attr.getValues<float>()) {
    // TODO(arash): There is probably a better way to do this?
    if (std::abs((std::abs(value) - 1.0f)) > 0.005f) return false;
  }
  return true;
}

#include "larq_compute_engine/mlir/transforms/generated_prepare.inc"

void PrepareLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  // Cleanup dead ops manually. LCE ops are not registered to the TF dialect so
  // op->hasNoSideEffect() will return false. Therefor applyPatternsGreedily
  // won't automatically remove the dead nodes. See
  // https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/IR/Operation.h#L457-L462
  patterns.insert<mlir::CleanupDeadOps<TF::LqceBsignOp>,
                  mlir::CleanupDeadOps<TF::LqceBconv2d64Op>>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareLCEPass() {
  return std::make_unique<PrepareLCE>();
}

static PassRegistration<PrepareLCE> pass("tfl-prepare-lce", "Inject LCE Ops");

}  // namespace TFL
}  // namespace mlir
