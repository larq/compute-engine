#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "llvm/ADT/Optional.h"
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
  auto filter_type = filter.getType().cast<ShapedType>();
  return filter_type.getElementType().isF32() &&
         filter_type.getShape().size() == 4;
}

DenseElementsAttr ComputeBSignAndExpandTo4D(Attribute attr) {
  auto tensor = attr.cast<DenseElementsAttr>();
  auto channels = tensor.getNumElements();
  auto tensor_type = attr.getType().cast<ShapedType>();

  std::vector<APFloat> results;
  results.reserve(channels);
  for (auto value : tensor.getValues<float>()) {
    auto result = std::signbit(value) ? -1.0f : 1.0f;
    results.push_back(APFloat(result));
  }

  auto expanded_shape =
      RankedTensorType::get({channels, 1, 1, 1}, tensor_type.getElementType());
  return DenseElementsAttr::get(expanded_shape, results);
}

bool HasNegativeValues(Attribute attr) {
  if (!attr.isa<DenseElementsAttr>()) return false;

  for (auto value : attr.cast<DenseElementsAttr>().getValues<float>()) {
    if (value < 0.0f) return true;
  }
  return false;
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
  std::vector<float> old_values(num_rows * unpacked_channels);

  int i = 0;
  for (float x : dense_elements_iter) {
    old_values[i++] = x;
  }
  assert(i == num_rows * unpacked_channels);

  using namespace compute_engine::core;
  packbits_matrix<BitpackOrder::Canonical>(
      old_values.data(), num_rows, unpacked_channels, new_values.data());

  RankedTensorType out_tensor_type =
      RankedTensorType::get({shape[0], shape[1], shape[2], packed_channels},
                            builder.getIntegerType(bitwidth));

  return DenseElementsAttr::get<PackedType>(out_tensor_type, new_values);
}

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"

llvm::Optional<RankedTensorType> maybeGetBitpackedType(
    PatternRewriter& rewriter, ShapedType existing_type) {
  if (existing_type.getElementType().isInteger(32)) return llvm::None;

  const auto existing_shape = existing_type.getShape();
  if (existing_shape.size() != 4) return llvm::None;

  const auto channels = existing_shape[3];
  const auto packed_channels = (channels + 32 - 1) / 32;
  return RankedTensorType::get({existing_shape[0], existing_shape[1],
                                existing_shape[2], packed_channels},
                               rewriter.getIntegerType(32));
}

template <typename BinaryOp>
struct SetBitpackedActivations : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp outer_binary_op,
                                PatternRewriter& rewriter) const override {
    Operation* input_op = outer_binary_op.input().getDefiningOp();

    // If the input op has more than one use, we can't apply an optimisation.
    if (!input_op || !input_op->hasOneUse()) return failure();

    // Try and match `input_op` to a binary convolution.
    auto inner_bconv_op = dyn_cast_or_null<TF::LceBconv2dOp>(input_op);
    if (inner_bconv_op) {
      if (inner_bconv_op.padding() == "SAME" &&
          inner_bconv_op.pad_values() != 1) {
        return failure();
      }

      if (auto maybe_bitpacked_type = maybeGetBitpackedType(
              rewriter, inner_bconv_op.getType().cast<ShapedType>())) {
        rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
            inner_bconv_op, *maybe_bitpacked_type, inner_bconv_op.getOperands(),
            inner_bconv_op.getAttrs());
        return success();
      }
    }

    // Otherwise, try and match `input_op` to a maxpool.
    auto maxpool_op = dyn_cast_or_null<TFL::MaxPool2DOp>(input_op);
    if (maxpool_op) {
      if (maxpool_op.fused_activation_function() != "NONE") {
        return failure();
      }

      if (auto bitpacked_type = maybeGetBitpackedType(
              rewriter, maxpool_op.getType().cast<ShapedType>())) {
        rewriter.replaceOpWithNewOp<TF::LceBMaxPool2dOp>(
            maxpool_op, *bitpacked_type, maxpool_op.input(),
            rewriter.getStringAttr(maxpool_op.padding()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.stride_h()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.stride_w()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.filter_width()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.filter_height()));
        return success();
      }
    }

    return failure();
  };
};

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  if (experimental_enable_bitpacked_activations_) {
    patterns.insert<SetBitpackedActivations<TF::LceBconv2dOp>>(ctx);
    patterns.insert<SetBitpackedActivations<TF::LceBMaxPool2dOp>>(ctx);
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
