#ifndef LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_
#define LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "larq_compute_engine/mlir/ir/lce_dialect.h.inc"
// clang-format on

#include "larq_compute_engine/mlir/ir/lce_enum.h.inc"

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_ops.h.inc"

#endif  // LARQ_COMPUTE_ENGINE_MLIR_IR_LCE_OPS_H_
