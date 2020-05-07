#ifndef COMPUTE_ENGINE_CORE_PACKBITS_UTILS_H_
#define COMPUTE_ENGINE_CORE_PACKBITS_UTILS_H_

#include "larq_compute_engine/core/packbits.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;
namespace core {

template <typename TBitpacked>
int GetPackedTensorSize(const RuntimeShape& shape) {
  constexpr auto bitwidth = std::numeric_limits<
      typename std::make_unsigned<TBitpacked>::type>::digits;
  const int dims = shape.DimensionsCount();
  // Pack the tensor along the last dimension
  const int rows = FlatSizeSkipDim(shape, dims - 1);
  const int cols = shape.Dims(dims - 1);
  return ce::core::GetPackedMatrixSize(rows, cols, bitwidth);
}

// Convenience function for bitpacking a tensor along its last dimension
// and updating the tensor shape
template <BitpackOrder bitpack_order, class T, class TBitpacked>
inline void packbits_tensor(const RuntimeShape& in_shape, const T* in_data,
                            const std::int32_t zero_point,
                            RuntimeShape& out_shape, TBitpacked* out_data) {
  const int dims = in_shape.DimensionsCount();
  // Pack the tensor along the last dimension
  const int rows = FlatSizeSkipDim(in_shape, dims - 1);
  const int cols = in_shape.Dims(dims - 1);

  {
    ruy::profiler::ScopeLabel label("Packbits");
    ce::core::packbits_matrix<bitpack_order>(in_data, rows, cols, out_data,
                                             zero_point);
  }

  out_shape.ReplaceWith(dims, in_shape.DimsData());
  out_shape.SetDim(dims - 1, GetPackedSize<TBitpacked>(cols));
}

// Convenience function for going from a shape to the packed shape
template <class TBitpacked>
RuntimeShape packed_shape(const RuntimeShape& in_shape) {
  constexpr auto bitwidth = std::numeric_limits<TBitpacked>::digits;
  const int dims = in_shape.DimensionsCount();
  RuntimeShape out_shape(in_shape);
  out_shape.SetDim(dims - 1,
                   (in_shape.Dims(dims - 1) + bitwidth - 1) / bitwidth);
  return out_shape;
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_PACKBITS_UTILS_H_
