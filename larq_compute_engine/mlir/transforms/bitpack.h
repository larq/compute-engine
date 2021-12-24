#ifndef LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_BITPACK_H_
#define LARQ_COMPUTE_ENGINE_MLIR_TRANSFORMS_BITPACK_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace TFL {

DenseElementsAttr Bitpack(mlir::Builder* builder, Attribute x);

DenseElementsAttr Unpack(Attribute x, ShapedType result_type);

}  // namespace TFL
}  // namespace mlir

#endif
