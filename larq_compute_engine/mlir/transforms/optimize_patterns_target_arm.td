include "larq_compute_engine/mlir/transforms/optimize_patterns_common.td"

// Insert a binary maxpool if a maxpool is followed by a sign op.
def : Pat<(LQ_QuantizeOp
              (TFL_MaxPool2DOp:$pool_output
                  $input,
                  $padding,
                  $stride_w,
                  $stride_h,
                  $filter_width,
                  $filter_height,
                  $fused_activation_function)),
          (LQ_BMaxPool2dOp
              (LQ_QuantizeOp $input),
              $padding,
              $stride_w,
              $stride_h,
              $filter_width,
              $filter_height),
          [(HasOneUse $pool_output)]>;
