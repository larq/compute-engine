#include "larq_compute_engine/mlir/ir/lce_ops.h"

// Static initialization for Larq Compute Engine op registration.
static mlir::DialectRegistration<mlir::TF::LarqDialect> lce_ops;
