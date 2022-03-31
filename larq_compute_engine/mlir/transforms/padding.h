#ifndef LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_PADDING_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_PADDING_H_

#include "larq_compute_engine/core/types.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace TFL {

inline DenseElementsAttr GetValidPadAttr(Attribute paddings_attr) {
  if (!paddings_attr.isa<DenseElementsAttr>()) return nullptr;
  auto paddings = paddings_attr.cast<DenseElementsAttr>();
  // The shape should be [4,2]
  auto pad_type = paddings.getType();
  if (pad_type.getRank() != 2) return nullptr;
  auto pad_shape = pad_type.getShape();
  if (pad_shape[0] != 4 || pad_shape[1] != 2) return nullptr;
  return paddings;
}

using ShapeRefType = ::llvm::ArrayRef<int64_t>;

inline ShapeRefType GetShape4D(Value tensor) {
  auto tensor_type = tensor.getType().dyn_cast<RankedTensorType>();
  if (!tensor_type) return ShapeRefType();
  ShapeRefType tensor_shape = tensor_type.getShape();
  if (tensor_shape.size() != 4) return ShapeRefType();
  return tensor_shape;
}

inline bool IsSamePadding1D(DenseElementsAttr paddings, uint64_t dimension,
                            int input_size, int output_size, int stride) {
  using compute_engine::core::CeilDiv;
  int pad_before = paddings.getValues<int>()[]{dimension, 0}];
  int pad_after = paddings.getValues<int>()[]{dimension, 1}];
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
