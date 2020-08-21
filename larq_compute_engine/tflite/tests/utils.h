#ifndef LARQ_COMPUTE_ENGINE_TFLITE_TESTS_UTILS
#define LARQ_COMPUTE_ENGINE_TFLITE_TESTS_UTILS

#include <cstdint>
#include <random>
#include <string>

#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

constexpr int Padding_ONE = Padding_MAX + 1;

const char* GetPaddingName(enum Padding padding) {
  switch (padding) {
    case Padding_VALID:
      return "VALID";
    case Padding_SAME:
      return "SAME";
    default:
      return "UNKNOWN";
  };
}

std::string getActivationString(const enum ActivationFunctionType activation) {
  if (activation == ActivationFunctionType_RELU) {
    return "RELU";
  } else if (activation == ActivationFunctionType_NONE) {
    return "NONE";
  }
  return "UNKOWN";
}

// Useful struct, in particular for int8 quantization
template <typename T>
struct LceTensor : public TensorData {
  LceTensor(std::vector<int> shape = {})
      : TensorData(GetTensorType<T>(), shape), reciprocal_scale(1){};

  template <typename TOther>
  void SetQuantizationParams(const LceTensor<TOther>& rhs) {
    reciprocal_scale = rhs.reciprocal_scale;
    scale = rhs.scale;
    zero_point = rhs.zero_point;
  }

  // For int8, `int8_value` has range [-128, 127]
  //     real_value = scale * (int8_value - zero_point)
  // To represent {-1,+1} without error, we require
  //     int_value = zero_point + (+-1) / scale
  // Therefore `scale` must be of the form 1/n
  // and we require
  //     abs(zero_point) + abs(1/scale) <= 127
  // Note that filter.zero_point = 0 for Conv2D,
  // and BConv2D does not have int8 filters.

  template <typename Gen>
  void GenerateQuantizationParams(Gen& gen) {
    reciprocal_scale = std::uniform_int_distribution<>(1, 20)(gen);
    scale = 1.0f / float(reciprocal_scale);
    zero_point = std::uniform_int_distribution<>(-20, 20)(gen);
  }

  // This has zero_point = 0
  template <typename Gen>
  void GenerateQuantizationParamsPerChannel(Gen& gen) {
    reciprocal_scale = std::uniform_int_distribution<>(1, 20)(gen);
    // Do not set this->scale because TF Lite might expect it to be 0 when
    // per_channel_quantization is used.
    float s = 1.0f / float(reciprocal_scale);

    per_channel_quantization = true;
    std::size_t count = shape[channel_index];
    per_channel_quantization_scales = std::vector<float>(count, s);
    per_channel_quantization_offsets = std::vector<std::int64_t>(count, 0);
  }

  T Quantize(const std::int32_t x) const {
    if (std::is_floating_point<T>::value) {
      return x;
    } else {
      std::int32_t y = zero_point + reciprocal_scale * x;
      y = std::min<std::int32_t>(y, std::numeric_limits<T>::max());
      y = std::max<std::int32_t>(y, std::numeric_limits<T>::lowest());
      return y;
    }
  }

  T Quantize(const float x) const {
    if (std::is_floating_point<T>::value) {
      return x;
    } else {
      std::int32_t y = zero_point + std::roundl(x / scale);
      y = std::min<std::int32_t>(y, std::numeric_limits<T>::max());
      y = std::max<std::int32_t>(y, std::numeric_limits<T>::lowest());
      return y;
    }
  }

  float DeQuantize(const T& x) const { return scale * (x - zero_point); }

  // For floating-point tests, this will also work because
  // then zero_point = 0 and reciprocal_scale = 1
  template <typename Gen, typename Iter>
  void GenerateSigns(Gen& gen, const Iter& begin, const Iter& end) const {
    auto sign_gen = [&gen, this]() {
      return std::bernoulli_distribution(0.5)(gen) ? Quantize(1) : Quantize(-1);
    };
    std::generate(begin, end, sign_gen);
  }

  // This is 1/scale
  std::int32_t reciprocal_scale;
};

//
// Utility functions to switch between RuntimeShape and std::vector<int>
//

std::vector<int> GetShape(const RuntimeShape& shape) {
  std::vector<int> s(shape.DimensionsCount());
  for (int i = 0; i < shape.DimensionsCount(); ++i) {
    s[i] = shape.Dims(i);
  }
  return s;
}

RuntimeShape GetShape(const std::vector<int>& shape) {
  RuntimeShape s;
  s.BuildFrom(shape);
  return s;
}

}  // namespace tflite

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_TESTS_UTILS
