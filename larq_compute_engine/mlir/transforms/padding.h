#ifndef LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_PADDING_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_PADDING_H_

#include "larq_compute_engine/core/types.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace TFL {

inline bool IsSamePadding1D(DenseElementsAttr paddings, uint64_t dimension,
                            int input_size, int output_size, int stride) {
  using compute_engine::core::CeilDiv;
  int pad_before = paddings.getValue<int>({dimension, 0});
  int pad_after = paddings.getValue<int>({dimension, 1});
  const int pad_total = pad_before + pad_after;
  return (output_size == CeilDiv(input_size, stride)) &&
         (pad_before == (pad_total / 2)) &&
         (pad_after == ((pad_total + 1) / 2));
}

inline bool IsNoPadding(DenseElementsAttr paddings, uint64_t dimension) {
  return paddings.getValue<int>({dimension, 0}) == 0 &&
         paddings.getValue<int>({dimension, 1}) == 0;
}

}  // namespace TFL
}  // namespace mlir

#endif
