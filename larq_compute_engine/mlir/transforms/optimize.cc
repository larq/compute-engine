#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public FunctionPass<OptimizeLCE> {
 public:
  // The default value must be true so that we can run with the optimisation in
  // the file-check tests.
  explicit OptimizeLCE() : experimental_enable_bitpacked_activations_(true) {}
  explicit OptimizeLCE(bool experimental_enable_bitpacked_activations)
      : experimental_enable_bitpacked_activations_(
            experimental_enable_bitpacked_activations) {}
  void runOnFunction() override;

 private:
  bool experimental_enable_bitpacked_activations_;
};

bool IsConstantValue(Attribute values, float expected_value) {
  if (!values.isa<DenseElementsAttr>()) return false;

  for (auto value : values.cast<DenseElementsAttr>().getValues<float>()) {
    if (value != expected_value) return false;
  }
  return true;
}

bool IsConv2DFilter(Attribute filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;
  if (filter.getType().cast<ShapedType>().getShape().size() != 4) return false;
  return true;
}

DenseElementsAttr Bitpack(PatternRewriter& builder, Attribute x) {
  const auto& dense_elements_iter =
      x.cast<DenseElementsAttr>().getValues<float>();

  using PackedType = std::uint32_t;
  constexpr int bitwidth = std::numeric_limits<PackedType>::digits;

  auto shape = x.getType().cast<ShapedType>().getShape();
  int num_rows = shape[0] * shape[1] * shape[2];
  int unpacked_channels = shape[3];
  int packed_channels = (unpacked_channels + bitwidth - 1) / bitwidth;

  std::vector<PackedType> new_values(num_rows * packed_channels);

  const float* in_ptr = &(*dense_elements_iter.begin());
  using namespace compute_engine::core;
  packbits_matrix<BitpackOrder::Canonical>(in_ptr, num_rows, unpacked_channels,
                                           new_values.data());

  RankedTensorType out_tensor_type =
      RankedTensorType::get({shape[0], shape[1], shape[2], packed_channels},
                            builder.getIntegerType(bitwidth));

  return DenseElementsAttr::get<PackedType>(out_tensor_type, new_values);
}

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"

struct SetBconvReadWriteBitpacked : public OpRewritePattern<TF::LceBconv2dOp> {
  using OpRewritePattern<TF::LceBconv2dOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::LceBconv2dOp bconv_op,
                                     PatternRewriter& rewriter) const override {
    Value bconv_input = bconv_op.input();
    if (!bconv_input.hasOneUse()) return matchFailure();

    auto bconv_input_op =
        dyn_cast_or_null<TF::LceBconv2dOp>(bconv_input.getDefiningOp());
    if (!bconv_input_op) return matchFailure();

    const auto inner_tensor_type = bconv_input_op.getType().cast<ShapedType>();

    // We use 32-bit bitpacking.
    constexpr int bitwidth = 32;

    // We can only apply this transformation if the inner tensor type is F32.
    if (inner_tensor_type.getElementType().isInteger(bitwidth))
      return matchFailure();

    if (bconv_input_op.padding() == "SAME" && bconv_input_op.pad_values() != 1)
      return matchFailure();

    const auto inner_tensor_shape = inner_tensor_type.getShape();
    if (inner_tensor_shape.size() != 4) return matchFailure();

    const auto channels = inner_tensor_shape[3];
    const auto packed_channels = (channels + bitwidth - 1) / bitwidth;

    RankedTensorType new_inner_tensor_type =
        RankedTensorType::get({inner_tensor_shape[0], inner_tensor_shape[1],
                               inner_tensor_shape[2], packed_channels},
                              rewriter.getIntegerType(bitwidth));

    rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
        bconv_input_op, new_inner_tensor_type, bconv_input_op.input(),
        bconv_input_op.filter(), bconv_input_op.post_activation_multiplier(),
        bconv_input_op.post_activation_bias(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                bconv_input_op.channels_in()),
        bconv_input_op.strides(),
        rewriter.getStringAttr(bconv_input_op.padding()),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                bconv_input_op.pad_values()),
        rewriter.getStringAttr(bconv_input_op.data_format()),
        bconv_input_op.dilations(),
        rewriter.getStringAttr(bconv_input_op.filter_format()),
        rewriter.getStringAttr(bconv_input_op.activation()));

    rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
        bconv_op, bconv_op.getType(), bconv_op.input(), bconv_op.filter(),
        bconv_op.post_activation_multiplier(), bconv_op.post_activation_bias(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                bconv_op.channels_in()),
        bconv_op.strides(), rewriter.getStringAttr(bconv_op.padding()),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                bconv_op.pad_values()),
        rewriter.getStringAttr(bconv_op.data_format()), bconv_op.dilations(),
        rewriter.getStringAttr(bconv_op.filter_format()),
        rewriter.getStringAttr(bconv_op.activation()));

    return matchSuccess();
  };
};

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  if (experimental_enable_bitpacked_activations_) {
    patterns.insert<SetBconvReadWriteBitpacked>(ctx);
  }
  // Cleanup dead ops manually. LCE ops are not registered to the TF dialect so
  // op->hasNoSideEffect() will return false. Therefor applyPatternsGreedily
  // won't automatically remove the dead nodes. See
  // https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/IR/Operation.h#L457-L462
  patterns.insert<mlir::CleanupDeadOps<TF::LceBconv2dOp>>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizeLCEPass(
    bool experimental_enable_bitpacked_activations) {
  return std::make_unique<OptimizeLCE>(
      experimental_enable_bitpacked_activations);
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
