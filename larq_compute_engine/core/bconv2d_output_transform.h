#ifndef COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
#define COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace compute_engine {

namespace core {

using compute_engine::core::TBitpacked;

enum class OutputTransformDetails {
  Default,
  Preprocessed,
  PreprocessedIntegerOnly
};

// The `OutputTransform` struct describes what needs to be done to get from the
// int32 accumulator value to the final result that is written back to memory.
//
// This transform depends on the destination type `DstScalar` which can be:
// - float
// - TBitpacked (i.e. int32_t, meaning bitpacked output)
// - int8
template <typename AccumScalar, typename DstScalar,
          OutputTransformDetails = OutputTransformDetails::Default>
struct OutputTransform {};

// A part of the transformation is common to all output types.
// This part is described by `OutputTransformBase`.
template <typename AccumScalar>
struct OutputTransformBase {
  AccumScalar backtransform_add = 0;
  std::int32_t clamp_min = std::numeric_limits<AccumScalar>::lowest();
  std::int32_t clamp_max = std::numeric_limits<AccumScalar>::max();

  inline AccumScalar RunBase(const AccumScalar accum) const {
    // Backtransform can still be done in int32
    AccumScalar x = backtransform_add - 2 * accum;
    // Activation function can also be done in int32
    x = std::min<AccumScalar>(x, clamp_max);
    x = std::max<AccumScalar>(x, clamp_min);
    return x;
  }
};

// Output transformation for float kernels
template <typename AccumScalar>
struct OutputTransform<AccumScalar, float, OutputTransformDetails::Default>
    : OutputTransformBase<AccumScalar> {
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;

  float Run(const AccumScalar accum, int out_channel) const {
    TF_LITE_ASSERT(post_activation_multiplier != nullptr);
    TF_LITE_ASSERT(post_activation_bias != nullptr);
    // Post multiply and add are done in float
    float x = static_cast<float>(this->RunBase(accum));
    x *= post_activation_multiplier[out_channel];
    x += post_activation_bias[out_channel];
    return x;
  }
};

// Output transformation for bitpacked output
template <typename AccumScalar>
struct OutputTransform<AccumScalar, TBitpacked,
                       OutputTransformDetails::Default> {
  const AccumScalar* thresholds = nullptr;

  bool Run(const AccumScalar accum, int out_channel) const {
    TF_LITE_ASSERT(thresholds != nullptr);
    return accum > thresholds[out_channel];
  }
};

// Output transformation for 8-bit quantization
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int8_t,
                       OutputTransformDetails::Default>
    : OutputTransformBase<AccumScalar> {
  // These effective values are the post-activation multipliers and biases
  // divided by output_scale and including the output zero_point
  const float* effective_post_activation_multiplier = nullptr;
  const float* effective_post_activation_bias = nullptr;

  std::int8_t Run(const AccumScalar accum, int out_channel) const {
    TF_LITE_ASSERT(effective_post_activation_multiplier != nullptr);
    TF_LITE_ASSERT(effective_post_activation_bias != nullptr);
    // First convert to full precision to do the linear transformation
    float result_fp = static_cast<float>(this->RunBase(accum));
    result_fp *= effective_post_activation_multiplier[out_channel];
    result_fp += effective_post_activation_bias[out_channel];
    // Now round back to int32
    AccumScalar result = tflite::TfLiteRound(result_fp);
    // Clamp to int8 range
    result =
        std::min<AccumScalar>(result, std::numeric_limits<std::int8_t>::max());
    result = std::max<AccumScalar>(result,
                                   std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(result);
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
