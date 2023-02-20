#include <cmath>

#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/common.h"
#include "larq_compute_engine/mlir/transforms/passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

// Optimize LCE operations in functions.
struct OptimizeLCE
    : public PassWrapper<OptimizeLCE, OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const final { return "tfl-optimize-lce"; }
  llvm::StringRef getDescription() const final {
    return "Optimize within the TensorFlow Lite dialect";
  }
  OptimizeLCE() = default;
  OptimizeLCE(const OptimizeLCE& pass) {}
  OptimizeLCE(const LCETarget target) { target_.setValue(target); }

  void runOnOperation() override;

 private:
  Option<LCETarget> target_{
      *this, "target", llvm::cl::desc("Platform to target."),
      llvm::cl::values(clEnumValN(LCETarget::ARM, "arm", "ARM target"),
                       clEnumValN(LCETarget::XCORE, "xcore", "XCORE target"))};
};

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
 * The function below accepts the constants a, c, C, a vector of 𝛄, and a
 * vector of 𝛃. It writes the computed thresholds into the vector `thresholds`.
 * It is assumed that when these computed thresholds are used, the filter
 * weights are first multiplied by the sign of corresponding channel multiplier
 * (this performs the weight flipping that is required in the 𝛄 < 0 case). The
 * output of the binary convolution using these thresholds *will not* be correct
 * if this does not happen! Currently, this filter weight flipping is part of
 * the rewrite pattern `WriteBitpackedActivationsPat` in
 * `optimize_patterns_common.td`.
 */
void ComputeWriteBitpackedOutputThresholds(
    Builder& builder, const float backtransform_add, const float clamp_min,
    const float clamp_max, const DenseElementsAttr& multipliers,
    const DenseElementsAttr& biases, std::vector<Attribute>& thresholds) {
  assert(thresholds.size() == 0);

  const auto element_type = builder.getIntegerType(32);

  constexpr std::int32_t neg_inf = std::numeric_limits<std::int32_t>::min();
  constexpr std::int32_t pos_inf = std::numeric_limits<std::int32_t>::max();

  // Iterate through the multiplier/bias pairs and compute the thresholds.
  for (auto mult_bias_pair :
       llvm::zip(multipliers.getValues<float>(), biases.getValues<float>())) {
    float mult = std::get<0>(mult_bias_pair);
    float bias = std::get<1>(mult_bias_pair);

    // Check for the 𝛄 = 0 special case.
    if (mult == 0.0f) {
      if (bias < 0.0f) {
        thresholds.push_back(IntegerAttr::get(
            element_type, neg_inf));  // We need to always write a 1-bit.
      } else {
        thresholds.push_back(IntegerAttr::get(
            element_type, pos_inf));  // We need to always write a 0-bit.
      }
      continue;
    }

    float effective_clamp_min, effective_clamp_max;
    if (mult > 0.0f) {
      effective_clamp_min = clamp_min;
      effective_clamp_max = clamp_max;
    } else {
      effective_clamp_min = -1 * clamp_max;
      effective_clamp_max = -1 * clamp_min;
    }

    const float output_range_start =
        (effective_clamp_min * std::abs(mult) + bias);
    const float output_range_end =
        (effective_clamp_max * std::abs(mult) + bias);

    // Check for the clamping range not crossing zero special case.
    if (output_range_start < 0 && output_range_end < 0) {
      thresholds.push_back(IntegerAttr::get(
          element_type, neg_inf));  // We need to always write a 1-bit.
      continue;
    }
    if (output_range_start >= 0 && output_range_end >= 0) {
      thresholds.push_back(IntegerAttr::get(
          element_type, pos_inf));  // We need to always write a 0-bit.
      continue;
    }

    // The general case.
    thresholds.push_back(IntegerAttr::get(
        element_type,
        std::floor(0.5 * (bias / std::abs(mult) + backtransform_add))));
  }
}

DenseElementsAttr GetSignsOfVectorAndBroadcast4D(TypedAttr vector_attr) {
  const auto vector = vector_attr.cast<DenseElementsAttr>();
  const auto vector_type = vector_attr.getType().cast<ShapedType>();
  assert(vector_type.getShape().size() == 1);
  const auto vector_length = vector_type.getShape()[0];
  const auto element_type = vector_type.getElementType();

  std::vector<Attribute> signs(vector_length);
  for (std::size_t i = 0; i < vector_length; ++i) {
    const auto sign = vector.getValues<float>()[i] >= 0.0f ? 1.0f : -1.0f;
    signs[i] = FloatAttr::get(element_type, sign);
  }

  const RankedTensorType type =
      RankedTensorType::get({vector_length, 1, 1, 1}, element_type);
  return DenseElementsAttr::get(type, signs);
}

DenseElementsAttr GetBitpackedOutputThresholds(
    Builder& builder, TypedAttr filter_attr,
    Attribute post_activation_multiplier_attr,
    Attribute post_activation_bias_attr,
    Attribute fused_activation_function_attr) {
  const auto post_activation_multiplier =
      post_activation_multiplier_attr.cast<DenseElementsAttr>();
  const auto post_activation_bias =
      post_activation_bias_attr.cast<DenseElementsAttr>();
  const auto fused_activation_function =
      fused_activation_function_attr.cast<StringAttr>().getValue();

  // Compute the constants `backtransform_add` and `clamp_min/max`.
  const auto filter_type = filter_attr.getType().cast<ShapedType>();
  const auto filter_shape = filter_type.getShape();
  const std::int32_t backtransform_add =
      filter_shape[1] * filter_shape[2] * filter_shape[3];
  const auto clamp_min_max =
      llvm::StringSwitch<std::pair<std::int32_t, std::int32_t>>(
          fused_activation_function)
          .Case("RELU", {0, backtransform_add})
          .Case("RELU_N1_TO_1", {-1, 1})
          .Case("RELU6", {0, 6})
          .Default({-1 * backtransform_add, backtransform_add});
  const std::int32_t clamp_min = std::get<0>(clamp_min_max);
  const std::int32_t clamp_max = std::get<1>(clamp_min_max);

  std::vector<Attribute> thresholds;
  ComputeWriteBitpackedOutputThresholds(
      builder, static_cast<float>(backtransform_add),
      static_cast<float>(clamp_min), static_cast<float>(clamp_max),
      post_activation_multiplier, post_activation_bias, thresholds);

  const RankedTensorType type = RankedTensorType::get(
      {(std::int32_t)thresholds.size()}, builder.getIntegerType(32));
  return DenseElementsAttr::get(type, thresholds);
}

namespace optimize_target_arm {
#include "larq_compute_engine/mlir/transforms/generated_optimize_target_arm.inc"
}

namespace optimize_target_other {
#include "larq_compute_engine/mlir/transforms/generated_optimize_target_other.inc"
}

namespace optimize_bitpack_activations {
#include "larq_compute_engine/mlir/transforms/generated_bitpack_activations.inc"
}

void OptimizeLCE::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  if (target_ == LCETarget::ARM) {
    optimize_target_arm::populateWithGenerated(patterns);
  } else {
    optimize_target_other::populateWithGenerated(patterns);
  }
  optimize_bitpack_activations::populateWithGenerated(patterns);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateOptimizeLCEPass(
    const LCETarget target) {
  return std::make_unique<OptimizeLCE>(target);
}

static PassRegistration<OptimizeLCE> pass;

}  // namespace TFL
}  // namespace mlir
