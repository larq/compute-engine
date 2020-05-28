include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/transforms/quantize_patterns.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def : Pat<(LQ_Bconv2dOp
              $input,
              $filter,
              (TFL_DequantizeOp
                  (TFL_QuantizeOp $post_activation_multiplier, $qtype1)),
              (TFL_DequantizeOp
                  (TFL_QuantizeOp $post_activation_bias, $qtype2)),
              $output_threshold,
              $channels_in,
              $dilation_height_factor,
              $dilation_width_factor,
              $fused_activation_function,
              $pad_values,
              $padding,
              $stride_height,
              $stride_width),
          (LQ_Bconv2dOp $input,
              $filter,
              $post_activation_multiplier,
              $post_activation_bias,
              $output_threshold,
              $channels_in,
              $dilation_height_factor,
              $dilation_width_factor,
              $fused_activation_function,
              $pad_values,
              $padding,
              $stride_height,
              $stride_width)>;
