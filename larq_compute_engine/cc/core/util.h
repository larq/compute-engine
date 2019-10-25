#ifndef COMPUTE_ENGINE_CORE_UTIL_H
#define COMPUTE_ENGINE_CORE_UTIL_H

#include <array>

// Convenience functions for TF
#ifdef UTIL_TF
#include "tensorflow/core/framework/tensor.h"
#endif

// Convenience functions for TF lite
#ifdef UTIL_TFLITE
#include "tensorflow/lite/c/c_api_internal.h"
#endif

namespace compute_engine {
namespace core {

// Lightweight convenience struct to pass tensors
// It is similar to tensorflow::tensor and TfLiteTensor
// except that the number of dimensions is fixed at compile time
// as a template argument.
// It is meant for using internally in our functors.
template <typename T, size_t D>
struct Tensor {
  Tensor() : data(0){};
  Tensor(T* d, std::array<long int, D> s) : data(d), shape(s) {}

#ifdef UTIL_TF
  Tensor(const tensorflow::Tensor& t) : data((T*)(t.flat<T>().data())) {
      for (size_t i = 0; i < D && (signed)i < t.dims(); ++i) {
          shape[i] = t.dim_size(i);
      }
  }
#endif

#ifdef UTIL_TFLITE
  Tensor(const TfLiteTensor* t) : data((T*)(t->data.raw)) {
    for (size_t i = 0; i < D && i < t->dims->size; ++i) {
      shape[i] = t->dims->data[i];
    }
  }
#endif

  T* data;
  std::array<long int, D> shape;
};

}  // namespace core
}  // namespace compute_engine

#endif
