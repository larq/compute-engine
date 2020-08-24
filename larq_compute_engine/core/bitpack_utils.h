#ifndef COMPUTE_ENGINE_CORE_BITPACK_UTILS_H_
#define COMPUTE_ENGINE_CORE_BITPACK_UTILS_H_

#include "larq_compute_engine/core/bitpack.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;
namespace core {

inline int GetPackedTensorSize(const RuntimeShape& shape) {
  const int dims = shape.DimensionsCount();
  // Pack the tensor along the last dimension
  const int rows = FlatSizeSkipDim(shape, dims - 1);
  const int cols = shape.Dims(dims - 1);
  return ce::core::GetPackedMatrixSize(rows, cols);
}

// Convenience function for bitpacking a tensor along its last dimension
// and updating the tensor shape
template <class T>
inline void bitpack_tensor(const RuntimeShape& in_shape, const T* in_data,
                           const std::int32_t zero_point,
                           TBitpacked* out_data) {
  const int dims = in_shape.DimensionsCount();
  // Pack the tensor along the last dimension
  const int rows = FlatSizeSkipDim(in_shape, dims - 1);
  const int cols = in_shape.Dims(dims - 1);

  ce::core::bitpack_matrix(in_data, rows, cols, out_data, zero_point);
}

// Convenience function for going from a shape to the packed shape
inline RuntimeShape packed_shape(const RuntimeShape& in_shape) {
  const int dims = in_shape.DimensionsCount();
  RuntimeShape out_shape(in_shape);
  out_shape.SetDim(dims - 1, GetPackedSize(in_shape.Dims(dims - 1)));
  return out_shape;
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BITPACK_UTILS_H_
