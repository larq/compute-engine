#ifndef COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
#define COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/round.h"

namespace compute_engine {
namespace core {

// Parameters that are needed for both float and int8 kernels
template <typename AccumScalar>
struct OutputTransformBase {
  AccumScalar backtransform_add = 0;
  AccumScalar clamp_min = std::numeric_limits<AccumScalar>::lowest();
  AccumScalar clamp_max = std::numeric_limits<AccumScalar>::max();

  AccumScalar Run(const AccumScalar accum) const {
    // Backtransform can still be done in int32
    AccumScalar x = backtransform_add - 2 * accum;
    // Activation function can also be done in int32
    x = std::min<AccumScalar>(x, clamp_max);
    x = std::max<AccumScalar>(x, clamp_min);
    return x;
  }
};

// Although we don't technically need QuantizationFlavor,
// we leave it here for now because later we might need to distinguish between
// 8-bit bitpacked output and int8-quantized output.
using tflite::cpu_backend_gemm::QuantizationFlavor;

template <typename AccumScalar, typename ActivationScalar,
          QuantizationFlavor quantization_flavor =
              std::is_floating_point<ActivationScalar>::value
                  ? QuantizationFlavor::kFloatingPoint
                  : QuantizationFlavor::kIntegerWithPerRowMultiplier>
struct OutputTransform {};

// Parameters that are needed only for float kernels
template <typename AccumScalar>
struct OutputTransform<AccumScalar, float> : OutputTransformBase<AccumScalar> {
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;

  using Base = OutputTransformBase<AccumScalar>;

  float Run(const AccumScalar accum, int out_channel) const {
    // Post multiply and add are done in float
    float x = static_cast<float>(Base::Run(accum));
    if (post_activation_multiplier) {
      x *= post_activation_multiplier[out_channel];
    }
    if (post_activation_bias) {
      x += post_activation_bias[out_channel];
    }
    return x;
  }
};

// Kernels that write bitpacked output
// This can happen in both int8 models and float models
// TODO: For int8 models that have post_activation_ data in int8, this should be
// changed
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int32_t>
    : OutputTransform<AccumScalar, float> {
  bool Run(const AccumScalar accum, int out_channel) const {
    // Run the float version
    float x = OutputTransform<AccumScalar, float>::Run(accum, out_channel);
    return (x < 0);
  }
};

// OutputTransform for 8-bit quantization
//
// For devices that have an FPU, we can gain a lot of accuracy by doing a part
// of the output transform in float, and then converting back to 8-bit.
//
// On devices without FPU, this flag should be disabled and the kernel will be
// compiled without float ops in the output transform, at the cost of some
// accuracy.
#define LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT

// Parameters that are needed only for int8 kernels
template <typename AccumScalar>
struct OutputTransform<AccumScalar, std::int8_t>
    : OutputTransformBase<AccumScalar> {
#ifdef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;
  std::int32_t output_zero_point = 0;
#else
  const std::int32_t* output_multiplier = nullptr;
  const std::int32_t* output_shift = nullptr;
  const std::int32_t* output_zero_point = nullptr;
#endif

  using Base = OutputTransformBase<AccumScalar>;

  std::int8_t Run(const AccumScalar accum, int out_channel) const {
    AccumScalar x = Base::Run(accum);
#ifdef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
    float y = static_cast<float>(x);
    if (post_activation_multiplier) {
      y *= post_activation_multiplier[out_channel];
    }
    if (post_activation_bias) {
      y += post_activation_bias[out_channel];
    }
    x = tflite::TfLiteRound(y);
    x += output_zero_point;
#else
    // Divide by 'scale'
    x = tflite::MultiplyByQuantizedMultiplier(x, output_multiplier[out_channel],
                                              output_shift[out_channel]);
    // Add effective zero point
    x += output_zero_point[out_channel];
#endif
    // Clamp to int8 range
    x = std::min<std::int32_t>(x, std::numeric_limits<std::int8_t>::max());
    x = std::max<std::int32_t>(x, std::numeric_limits<std::int8_t>::lowest());
    return static_cast<std::int8_t>(x);
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_OUTPUT_TRANSFORM_H_
