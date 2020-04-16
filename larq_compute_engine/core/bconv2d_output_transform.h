#ifndef COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
#define COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/round.h"

namespace compute_engine {
namespace core {

//
// The `OutputTransform` struct describes what needs to be done to get from the
// int32 accumulator value to the final result that is written back to memory.
//
// This transform depends on the destination type `DstScalar` which can be:
// - float
// - int32 (meaning bitpacked output)
// - int8
template <typename AccumScalar, typename DstScalar>
struct OutputTransform {};

// A part of the transformation is common to all output types.
// This part is described by `OutputTransformBase`.
template <typename AccumScalar>
struct OutputTransformBase {
  AccumScalar backtransform_add = 0;
  AccumScalar clamp_min = std::numeric_limits<AccumScalar>::lowest();
  AccumScalar clamp_max = std::numeric_limits<AccumScalar>::max();

  AccumScalar RunBase(const AccumScalar accum) const {
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
struct OutputTransform<AccumScalar, float> : OutputTransformBase<AccumScalar> {
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;

  float Run(const AccumScalar accum, int out_channel) const {
    // Post multiply and add are done in float
    float x = static_cast<float>(this->RunBase(accum));
    if (post_activation_multiplier) {
      x *= post_activation_multiplier[out_channel];
    }
    if (post_activation_bias) {
      x += post_activation_bias[out_channel];
    }
    return x;
  }
};

// Output transformation for bitpacked output
// Currently uses an un-optimized path by using the float transform.
// TODO: Precompute a per-channel accumulation threshold
// so that we can simply do a single compare here:
//     return (accum <= threshold[out_channel]);
// Note: This would require modifying the converter or Prepare step to always
// have *positive* post_activation_multipliers, since otherwise we would have
// `<=` for some channels and `>=` for other channels.
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int32_t>
    : OutputTransform<AccumScalar, float> {
  bool Run(const AccumScalar accum, int out_channel) const {
    // Currently we take an un-optimized reference approach by first running the
    // float kernel and then taking the sign.
    float x = OutputTransform<AccumScalar, float>::Run(accum, out_channel);
    return (x < 0);
  }
};

// Output transformation for 8-bit quantization
//
// We use the following preprocessor flag which can be set in build scripts:
// LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
//
// For devices that have an FPU, we can gain accuracy by doing a part
// of the output transform in float, and then converting back to 8-bit.
//
// To not use the FPU, this flag should be disabled and the kernel will be
// compiled without float ops in the output transform, at the cost of some
// accuracy.

#ifdef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int8_t>
    : OutputTransformBase<AccumScalar> {
  // These effective values are the post-activation multipliers and biases
  // divided by output_scale
  const float* effective_post_activation_multiplier = nullptr;
  const float* effective_post_activation_bias = nullptr;
  std::int32_t output_zero_point = 0;

  std::int8_t Run(const AccumScalar accum, int out_channel) const {
    // First convert to full precision to do the linear transformation
    float result_fp = static_cast<float>(this->RunBase(accum));
    if (effective_post_activation_multiplier) {
      result_fp *= effective_post_activation_multiplier[out_channel];
    }
    if (effective_post_activation_bias) {
      result_fp += effective_post_activation_bias[out_channel];
    }
    // Now round back to int32
    AccumScalar result = tflite::TfLiteRound(result_fp);
    result += output_zero_point;
    // Clamp to int8 range
    result =
        std::min<std::int32_t>(result, std::numeric_limits<std::int8_t>::max());
    result = std::max<std::int32_t>(result,
                                    std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(result);
  }
};

#else   // LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT

template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int8_t>
    : OutputTransformBase<AccumScalar> {
  // output_multiplier and output_shift encode multiplication by
  // (post_activation_multiplier[channel] / scale)
  const std::int32_t* output_multiplier = nullptr;
  const std::int32_t* output_shift = nullptr;
  // output_effective_zero_point is addition by
  // (output_zero_point + round(post_activation_bias[channel] / scale)
  const std::int32_t* output_effective_zero_point = nullptr;

  std::int8_t Run(const AccumScalar accum, int out_channel) const {
    AccumScalar result = this->RunBase(accum);
    // Multiply by (post_activation_multiplier[channel] / scale)
    result = tflite::MultiplyByQuantizedMultiplier(
        result, output_multiplier[out_channel], output_shift[out_channel]);
    // Add post_activation_bias and output_zero_point
    result += output_effective_zero_point[out_channel];
    // Clamp to int8 range
    result =
        std::min<std::int32_t>(result, std::numeric_limits<std::int8_t>::max());
    result = std::max<std::int32_t>(result,
                                    std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(result);
  }
};
#endif  // LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
