#include <cmath>

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
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

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"

/**
 * =================================================
 * Computing thresholds for writing bitpacked output
 * =================================================
 *
 * Consider a single output element of a binary convolution, y. We have that:
 *
 *                       y = clamp(a - 2x, c, C) * 𝛄 + 𝛃
 *
 * where x is the xor-popcount accumulator,
 *       a is the backtransform addition (filter height * width * channels in),
 *       c is the clamp-min,
 *       C is the clamp-max,
 *       𝛄 is the channel multiplier,
 *       𝛃 is the channel bias.
 *
 * We want to write a 1-bit if and only if y < 0. To do this, we want to find
 * a threshold 𝛕 so that we can decide whether or not to write a 0-bit or a
 * 1-bit with a single comparison of x with 𝛕.
 *
 * -----------------------
 * The general case: 𝛄 > 0
 * -----------------------
 *
 * First, suppose that 𝛄 > 0.
 *
 * We can further assume that the clamping range crosses 0, i.e.
 * sign((c - 2x) * 𝛄 + 𝛃) != sign((C - 2x) * 𝛄 + 𝛃), so that the effect of
 * clamping can be ignored. (If this were not the case, the sign of y would be
 * constant and independent of x; we will consider this special case later.)
 *
 * We have that:
 *                   y < 0 <-> (a - 2x) * 𝛄 + 𝛃 < 0
 *                         <-> x > 0.5 * (𝛃 / 𝛄 + a)      (as 𝛄 > 0)
 *                         <-> x > ⌊0.5 * (𝛃 / 𝛄 + a)⌋    (as x is an integer)
 *
 * We can therefore use a threshold 𝛕 := ⌊0.5 * (𝛃 / 𝛄 + a)⌋, and write a 1-bit
 * if and only if x > 𝛕.
 *
 * -------------------
 * Special case: 𝛄 < 0
 * -------------------
 *
 * If we followed the same method as above with negative 𝛄 we would end up with
 * a threshold 𝛕, but the sign of the comparison would need to be flipped.
 *
 * However, we can use a trick to make 𝛄 positive. Let x' be xor-accumulator
 * after flipping all of the weights of the convolution (for this specific
 * channel). It's clear that x' = (a - x). Then we have that:
 *
 *                   y = clamp(a - 2x, c, C) * 𝛄 + 𝛃
 *                     = -1 * clamp(a - 2x, c, C) * (-𝛄) + 𝛃
 *                     = clamp(-1 * (a - 2x), -C, -c) * (-𝛄) + 𝛃
 *                     = clamp(2x - a, -C, -c) * (-𝛄) + 𝛃
 *                     = clamp(a - 2(a - x), -C, -c) * (-𝛄) + 𝛃
 *                     = clamp(a - 2x', -C, -c) * (-𝛄) + 𝛃
 *
 * So after flipping the weights for this channel, flipping the sign of 𝛄, and
 * flipping the signs of the clamping constants, we can apply the general case
 * from above to obtain a threshold 𝛕, and write a 1-bit if and only if x > 𝛕.
 *
 * ---------------------------------
 * Special case: sign(y) is constant
 * ---------------------------------
 *
 * As mentioned in the general case above, if the clamping range doesn't cross
 * zero then the sign of y is constant, and we will always write a 1-bit (or
 * always write a 0-bit). The same is true if 𝛄 = 0 (as then y = 𝛃 is constant).
 *
 * To always write a 1-bit (if and only if y < 0), we can set 𝛕 := -∞ (the
 * minimum signed representable number). Then, x > 𝛕 will always be true.
 *
 * To always write a 0-bit (if and only if y >= 0), we can set 𝛕 := ∞ (the
 * maximum signed representable number). Then, x > 𝛕 will always be false.
 *
 * --------------
 * Implementation
 * --------------
 *
 * The function below accepts the constants a, c, C, a vector of 𝛄, and a vector
 * of 𝛃. It writes the computed thresholds into the vector `thresholds`, and
 * writes -1 or 1 into the vector `filter_per_channel_multipliers` to either
 * flip or not flip the convolution weights, as required if 𝛄 < 0.
 */
void ComputeWriteBitpackedOutputThresholds(
    const float backtransform_add, const float clamp_min, const float clamp_max,
    const DenseElementsAttr& multipliers, const DenseElementsAttr& biases,
    std::vector<std::int32_t>& thresholds,
    std::vector<std::int32_t>& filter_per_channel_multipliers) {
  assert(thresholds.size() == 0 && filter_per_channel_multipliers.size() == 0);

  constexpr std::int32_t neg_inf = std::numeric_limits<std::int32_t>::min();
  constexpr std::int32_t pos_inf = std::numeric_limits<std::int32_t>::max();

  // Iterate through the multiplier/bias pairs and compute the thresholds.
  for (auto mult_bias_pair :
       llvm::zip(multipliers.getValues<float>(), biases.getValues<float>())) {
    float mult = std::get<0>(mult_bias_pair);
    float bias = std::get<1>(mult_bias_pair);

    // Check for the 𝛄 = 0 special case.
    if (mult == 0.0f) {
      filter_per_channel_multipliers.push_back(1.0f);
      if (bias < 0.0f) {
        thresholds.push_back(neg_inf);  // We need to always write a 1-bit.
      } else {
        thresholds.push_back(pos_inf);  // We need to always write a 0-bit.
      }
      continue;
    }

    float effective_clamp_min;
    float effective_clamp_max;
    if (mult > 0.0f) {
      effective_clamp_min = clamp_min;
      effective_clamp_max = clamp_max;
      filter_per_channel_multipliers.push_back(1.0f);
    } else {
      effective_clamp_min = -1 * clamp_max;
      effective_clamp_max = -1 * clamp_min;
      filter_per_channel_multipliers.push_back(-1.0f);
    }

    const float output_range_start =
        (effective_clamp_min * std::abs(mult) + bias);
    const float output_range_end =
        (effective_clamp_max * std::abs(mult) + bias);

    // Check for the clamping range not crossing zero special case.
    if (output_range_start < 0 && output_range_end < 0) {
      thresholds.push_back(neg_inf);  // We need to always write a 1-bit.
      continue;
    }
    if (output_range_start >= 0 && output_range_end >= 0) {
      thresholds.push_back(pos_inf);  // We need to always write a 0-bit.
      continue;
    }

    // The general case.
    thresholds.push_back(
        std::floor(0.5 * (bias / std::abs(mult) + backtransform_add)));
  }
}

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

// TODO: Move to TableGen once enabled by default
struct SetBitpackedActivations : public OpRewritePattern<TF::QuantizeOp> {
  using OpRewritePattern<TF::QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::QuantizeOp quantize_op,
                                PatternRewriter& rewriter) const override {
    Operation* input_op = quantize_op.getOperand().getDefiningOp();

    // If the input op has more than one use, we can't apply an optimisation.
    if (!input_op || !input_op->hasOneUse()) return failure();

    // Try and match `input_op` to a binary convolution.
    auto bconv_op = dyn_cast_or_null<TF::Bconv2dOp>(input_op);
    if (!bconv_op) return failure();

    // We can't apply this transform if the activation or filters are already
    // bitpacked, or with 'same zero' padding.
    const auto bconv_type = bconv_op.getType().cast<ShapedType>();
    const auto filter_type = bconv_op.filter().getType().cast<ShapedType>();
    const auto bitpacked_type = maybeGetBitpackedType(rewriter, bconv_type);
    if (!bconv_type.getElementType().isF32() ||
        !filter_type.getElementType().isF32() || !bitpacked_type ||
        (bconv_op.padding() == "SAME" && bconv_op.pad_values() != 1)) {
      return failure();
    }

    // As the inner bconv op will be writing bitpacked output, we need to
    // compute the thresholds for writing a 1-bit or a 0-bit.

    // Compute the constants `backtransform_add` and `clamp_min/max`.
    const auto filter_shape = filter_type.getShape();
    const std::int32_t backtransform_add =
        filter_shape[1] * filter_shape[2] * filter_shape[3];
    const auto clamp_min_max =
        llvm::StringSwitch<std::pair<std::int32_t, std::int32_t>>(
            bconv_op.fused_activation_function())
            .Case("RELU", {0, backtransform_add})
            .Case("RELU_N1_TO_1", {-1, 1})
            .Case("RELU6", {0, 6})
            .Default({-1 * backtransform_add, backtransform_add});
    const std::int32_t clamp_min = std::get<0>(clamp_min_max);
    const std::int32_t clamp_max = std::get<1>(clamp_min_max);

    // This is a nice trick taken from TF code: the `m_Constant` pattern
    // takes an `Attribute` as an argument. It tries to match a
    // constant-foldable `Value` and writes the value to the attribute if
    // the match succeeds.
    DenseElementsAttr filters;
    DenseElementsAttr multipliers;
    DenseElementsAttr biases;
    if (!matchPattern(bconv_op.filter(), m_Constant(&filters)) ||
        !matchPattern(bconv_op.post_activation_multiplier(),
                      m_Constant(&multipliers)) ||
        !matchPattern(bconv_op.post_activation_bias(), m_Constant(&biases))) {
      return failure();
    }

    std::vector<std::int32_t> thresholds;
    std::vector<std::int32_t> filter_per_channel_multipliers;
    ComputeWriteBitpackedOutputThresholds(
        static_cast<float>(backtransform_add), static_cast<float>(clamp_min),
        static_cast<float>(clamp_max), multipliers, biases, thresholds,
        filter_per_channel_multipliers);

    // Compute new filter values by folding in the per-channel multipliers.
    std::vector<float> new_filter_values(filter_shape[0] * filter_shape[1] *
                                         filter_shape[2] * filter_shape[3]);
    int i = 0;
    for (float value : filters.getValues<float>()) {
      float multiplier = filter_per_channel_multipliers[i / (filter_shape[1] *
                                                             filter_shape[2] *
                                                             filter_shape[3])];
      new_filter_values[i++] = value * multiplier;
    }

    // Create new inputs for the inner bconv op.
    Value filter_input = rewriter.create<ConstantOp>(
        bconv_op.filter().getLoc(),
        DenseElementsAttr::get<float>(filter_type, new_filter_values));
    RankedTensorType thresholds_type =
        RankedTensorType::get({filter_shape[0]}, rewriter.getIntegerType(32));
    Value thresholds_input = rewriter.create<ConstantOp>(
        bconv_op.output_threshold().getLoc(),
        DenseElementsAttr::get<std::int32_t>(thresholds_type, thresholds));

    // We need an empty input with which to overwrite the
    // `post_activation_multiplier` and `post_activation_bias` (which are no
    // longer needed, having computed the thresholds).
    Value empty_input = rewriter.create<ConstantOp>(
        bconv_op.getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());

    std::vector<Value> operands = {bconv_op.input(), filter_input, empty_input,
                                   empty_input, thresholds_input};
    rewriter.replaceOpWithNewOp<TF::Bconv2dOp>(quantize_op, *bitpacked_type,
                                               operands, bconv_op.getAttrs());

    return success();
  };
};

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  if (experimental_enable_bitpacked_activations_) {
    patterns.insert<SetBitpackedActivations>(ctx);
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
