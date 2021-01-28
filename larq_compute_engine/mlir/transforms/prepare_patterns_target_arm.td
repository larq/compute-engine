include "larq_compute_engine/mlir/transforms/prepare_patterns_common.td"

// On ARM we support 'same-zero' padding.
def : PrepareBConvPadValue0Pat<"SAME">;
