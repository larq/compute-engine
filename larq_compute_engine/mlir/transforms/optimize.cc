#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public PassWrapper<OptimizeLCE, FunctionPass> {
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

  LogicalResult matchAndRewrite(TF::LceBconv2dOp outer_bconv_op,
                                PatternRewriter& rewriter) const override {
    /*************
     * Match ops *
     *************/

    if (!outer_bconv_op.input().hasOneUse()) {
      return failure();
    }

    auto intermediate_maxpool_op = dyn_cast_or_null<TFL::MaxPool2DOp>(
        outer_bconv_op.input().getDefiningOp());

    if (intermediate_maxpool_op &&
        (!intermediate_maxpool_op.input().hasOneUse() ||
         intermediate_maxpool_op.fused_activation_function() != "NONE")) {
      return failure();
    }

    auto inner_bconv_op =
        intermediate_maxpool_op
            ? dyn_cast_or_null<TF::LceBconv2dOp>(
                  intermediate_maxpool_op.input().getDefiningOp())
            : dyn_cast_or_null<TF::LceBconv2dOp>(
                  outer_bconv_op.input().getDefiningOp());

    if (inner_bconv_op && inner_bconv_op.padding() == "SAME" &&
        inner_bconv_op.pad_values() != 1)
      return failure();

    if (!inner_bconv_op && !intermediate_maxpool_op) return failure();

    constexpr int bitwidth = 32;  // We use 32-bit bitpacking.

    /*********************
     * Compute new types *
     *********************/

    RankedTensorType new_inner_bconv_op_type;
    RankedTensorType new_intermediate_maxpool_op_type;

    // If applicable, compute the new type for the inner bconv op.
    if (inner_bconv_op) {
      ShapedType inner_bconv_op_type =
          inner_bconv_op.getType().cast<ShapedType>();

      // Don't apply this transformation if the type is already I32.
      if (inner_bconv_op_type.getElementType().isInteger(bitwidth)) {
        return failure();
      }

      // Compute the new type for the inner bconv op.
      const auto inner_bconv_op_shape = inner_bconv_op_type.getShape();
      if (inner_bconv_op_shape.size() != 4) return failure();
      const auto channels = inner_bconv_op_shape[3];
      const auto packed_channels = (channels + bitwidth - 1) / bitwidth;
      new_inner_bconv_op_type = RankedTensorType::get(
          {inner_bconv_op_shape[0], inner_bconv_op_shape[1],
           inner_bconv_op_shape[2], packed_channels},
          rewriter.getIntegerType(bitwidth));
    }

    // If applicable, compute the new type for the intermediate maxpool op.
    if (intermediate_maxpool_op) {
      ShapedType intermediate_maxpool_op_type =
          intermediate_maxpool_op.getType().cast<ShapedType>();

      // Don't apply this transformation if the type is already I32.
      if (intermediate_maxpool_op_type.getElementType().isInteger(bitwidth)) {
        return failure();
      }

      const auto intermediate_maxpool_op_shape =
          intermediate_maxpool_op_type.getShape();
      if (intermediate_maxpool_op_shape.size() != 4) return failure();
      const auto channels = intermediate_maxpool_op_shape[3];
      const auto packed_channels = (channels + bitwidth - 1) / bitwidth;
      new_intermediate_maxpool_op_type = RankedTensorType::get(
          {intermediate_maxpool_op_shape[0], intermediate_maxpool_op_shape[1],
           intermediate_maxpool_op_shape[2], packed_channels},
          rewriter.getIntegerType(bitwidth));
    }

    /***************************
     * Perform op replacements *
     ***************************/

    // We have to replace the inner-bconv op first...
    if (inner_bconv_op) {
      rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
          inner_bconv_op, new_inner_bconv_op_type, inner_bconv_op.getOperands(),
          inner_bconv_op.getAttrs());
    }
    // ...then the maxpool.
    if (intermediate_maxpool_op) {
      rewriter.replaceOpWithNewOp<TF::LceBMaxPool2dOp>(
          intermediate_maxpool_op, new_intermediate_maxpool_op_type,
          intermediate_maxpool_op.input(),
          rewriter.getStringAttr(intermediate_maxpool_op.padding()),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                  intermediate_maxpool_op.stride_h()),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                  intermediate_maxpool_op.stride_w()),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                  intermediate_maxpool_op.filter_width()),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                  intermediate_maxpool_op.filter_height()));
    }

    return success();
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
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeLCEPass(
    bool experimental_enable_bitpacked_activations) {
  return std::make_unique<OptimizeLCE>(
      experimental_enable_bitpacked_activations);
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
