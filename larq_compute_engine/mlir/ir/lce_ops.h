#ifndef LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_
#define LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

#include "larq_compute_engine/mlir/ir/lce_dialect.h.inc"

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_ops.h.inc"

}  // namespace TF
}  // namespace mlir

#endif  // LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_
