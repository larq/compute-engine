#include "larq_compute_engine/mlir/ir/lce_ops.h"

namespace mlir {
namespace TF {

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"

LarqDialect::LarqDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"
      >();
}
}  // namespace TF
}  // namespace mlir
