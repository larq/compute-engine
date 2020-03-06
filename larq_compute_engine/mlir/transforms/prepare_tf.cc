#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Prepare LCE operations in functions for subsequent legalization.
struct PrepareLCE : public FunctionPass<PrepareLCE> {
  void runOnFunction() override;
};

DenseElementsAttr GetConstantVector(Attribute filter, float val) {
  auto filter_type = filter.getType().cast<ShapedType>();
  auto filter_shape = filter_type.getShape();

  RankedTensorType type =
      RankedTensorType::get({filter_shape[3]}, filter_type.getElementType());
  return DenseElementsAttr::get(type, val);
}

bool IsBinaryFilter(Attribute filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;

  for (auto value : filter.cast<DenseElementsAttr>().getValues<float>()) {
    if (std::abs((std::abs(value) - 1.0f)) > 0.005f) return false;
  }
  return true;
}

bool IsSamePadding(Attribute paddings_attr, Value input, Value output,
                   ArrayAttr strides_attr) {
  if (!paddings_attr.isa<DenseElementsAttr>()) return false;
  auto paddings = paddings_attr.dyn_cast<DenseElementsAttr>();
  auto input_shape = input.getType().cast<RankedTensorType>().getShape();
  auto output_shape = output.getType().cast<RankedTensorType>().getShape();
  auto strides = strides_attr.getValue();

  int pad_height_left = paddings.getValue<int>({1, 0});
  int pad_height_right = paddings.getValue<int>({1, 1});
  int pad_width_left = paddings.getValue<int>({2, 0});
  int pad_width_right = paddings.getValue<int>({2, 1});

  int pad_height = pad_height_left + pad_height_right;
  int pad_width = pad_width_left + pad_width_right;

  int stride_height = strides[1].cast<IntegerAttr>().getInt();
  int stride_width = strides[2].cast<IntegerAttr>().getInt();

  return paddings.getValue<int>({0, 0}) == 0 &&
         paddings.getValue<int>({0, 1}) == 0 &&
         output_shape[1] ==
             (input_shape[1] + stride_height - 1) / stride_height &&
         output_shape[2] ==
             (input_shape[2] + stride_width - 1) / stride_width &&
         pad_height_left == pad_height / 2 &&
         pad_height_right == (pad_height + 1) / 2 &&
         pad_width_left == pad_width / 2 &&
         pad_width_right == (pad_width + 1) / 2 &&
         paddings.getValue<int>({3, 0}) == 0 &&
         paddings.getValue<int>({3, 1}) == 0;
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
