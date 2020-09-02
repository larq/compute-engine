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

/*
 * The `OutputTransform` struct describes what needs to be done to get from the
 * int32 accumulator value to the final result that is written back to memory.
 */
template <typename DstScalar,
          OutputTransformDetails = OutputTransformDetails::Default>
struct OutputTransform {};

/*
 * ------------
 * Float output
 * ------------
 *
 * Conceptually, the float output transform is as follows:
 *   0. We start with an XOR-popcount accumulator value `accum`. It lies in the
 *      range {0, ..., K}, where K := `kernel_height` * `kernel_width` *
 *      `channels_in` is the maximum number of bits that can be set.
 *   1. Perform a 'back-transformation', which is a linear transformation that
 *      maps {0, ..., K} onto {-K, ..., K}:
 *                         K - 2 * accum = -2 * accum + K
 *      This yields the 'true output' of the binary convolution operation, in
 *      the sense that it's the same value that you would get by running a
 *      normal floating-point convolution with +1/-1 input and +1/-1 weights.
 *   2. Perform a clamp operation that represents a fused activation function.
 *   3. Convert to float and perform a linear transformation with a multiplier
 *      and bias that represents a fused batch normalisation layer.
 *
 * Thus, the float output transform is:
 *                 float_cast(clamp(-2 * accum + K)) * mul + bias
 *
 * However, we can perform an equivalent, faster computation, by adjusting the
 * clamp values and fusing (part of) the back-transformation into the
 * multiplier/bias:
 *                 float_cast(clamp'(accum << 1)) * mul' + bias'
 *
 * Note that the left shift cannot be fused, because the left-shifted
 * accumulator value will either be always odd or always even, and we support
 * clamping to arbitrary integers.
 *
 * The parameters of the output transform are therefore two int32 clamp values
 * and float pointers to a multiplier and a bias.
 */
template <>
struct OutputTransform<float, OutputTransformDetails::Default> {
  std::int32_t clamp_min = std::numeric_limits<std::int32_t>::lowest();
  std::int32_t clamp_max = std::numeric_limits<std::int32_t>::max();
  const float* multiplier = nullptr;
  const float* bias = nullptr;

  float Run(const std::int32_t accum, int out_channel) const {
    TFLITE_DCHECK(multiplier != nullptr);
    TFLITE_DCHECK(bias != nullptr);
    std::int32_t x = accum << 1;
    x = std::max<std::int32_t>(std::min<std::int32_t>(x, clamp_max), clamp_min);
    return static_cast<float>(x) * multiplier[out_channel] + bias[out_channel];
  }
};

/*
 * -----------
 * Int8 output
 * -----------
 *
 * For int8 output, there are two additional conceptual steps:
 *   4. Multiply by the reciprocal of the int8 output scale, and add the int8
 *      zero-point.
 *   5. Round to an integer and clamp to the int8 range.
 *
 * We can fuse step (4) into the existing multiplier/bias, and so the int8
 * output transform is:
 *           int8_cast(float_cast(clamp'(accum << 1)) * mul + bias)
 *
 * Thus, the int8 output transform parameters are the same as in the float case.
 */
template <>
struct OutputTransform<std::int8_t, OutputTransformDetails::Default> {
  std::int32_t clamp_min = std::numeric_limits<std::int32_t>::lowest();
  std::int32_t clamp_max = std::numeric_limits<std::int32_t>::max();
  const float* multiplier = nullptr;
  const float* bias = nullptr;

  std::int8_t Run(const std::int32_t accum, int out_channel) const {
    TFLITE_DCHECK(multiplier != nullptr);
    TFLITE_DCHECK(bias != nullptr);
    // Clamping is done in int32
    std::int32_t x = accum << 1;
    x = std::max<std::int32_t>(std::min<std::int32_t>(x, clamp_max), clamp_min);
    // The linear transformation is done in float
    float y =
        static_cast<float>(x) * multiplier[out_channel] + bias[out_channel];
    // And then we round back to int32 and clamp to the int8 range
    std::int32_t z = tflite::TfLiteRound(y);
    z = std::min<std::int32_t>(z, std::numeric_limits<std::int8_t>::max());
    z = std::max<std::int32_t>(z, std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(z);
  }
};

/*
 * ----------------
 * Bitpacked output
 * ----------------
 *
 * For writing bitpacked output, the output transform needs to yield a single
 * bit - the sign of the result of conceptual step (3). In fact, it's possible
 * to avoid having to do the clamping and the linear transformation first, and
 * instead compare the accumulator directly against a pre-computed threshold to
 * work out which bit to return.
 *
 * Thus, the bitpacked output transform parameters are a single pointer to an
 * array of pre-computed thresholds.
 */
template <>
struct OutputTransform<TBitpacked, OutputTransformDetails::Default> {
  const std::int32_t* thresholds = nullptr;

  bool Run(const std::int32_t accum, int out_channel) const {
    TFLITE_DCHECK(thresholds != nullptr);
    return accum > thresholds[out_channel];
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
