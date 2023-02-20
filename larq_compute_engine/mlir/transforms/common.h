#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace TFL {

inline bool IsConstantValue(Attribute values, float expected_value) {
  if (!values.isa<DenseElementsAttr>()) return false;

  for (auto value : values.cast<DenseElementsAttr>().getValues<float>()) {
    if (value != expected_value) return false;
  }
  return true;
}

}  // namespace TFL
}  // namespace mlir
