#include "larq_compute_engine/cc/core/packbits.h"
#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

// Convenience function for bitpacking a tensor along its last dimension
// and updating the tensor shape
template <class T, class TBitpacked>
inline void packbits_tensor(const RuntimeShape& in_shape, const T* in_data,
                            RuntimeShape& out_shape,
                            std::vector<TBitpacked>& out_data,
                            const std::int32_t zero_point) {
  const int dims = in_shape.DimensionsCount();
  // Pack the tensor along the last dimension
  const int rows = FlatSizeSkipDim(in_shape, dims - 1);
  const int cols = in_shape.Dims(dims - 1);

  size_t rows_bp = 0, cols_bp = 0;
  size_t bitpadding = 0;
  {
    gemmlowp::ScopedProfilingLabel label("Packbits");
    ce::core::packbits_matrix(in_data, rows, cols, out_data, rows_bp, cols_bp,
                              bitpadding, ce::core::Axis::RowWise, zero_point);
  }

  out_shape.ReplaceWith(dims, in_shape.DimsData());
  out_shape.SetDim(dims - 1, cols_bp);
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

}  // namespace tflite
}  // namespace compute_engine
