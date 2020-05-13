include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/transforms/quantize_patterns.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def : Pat<(TF_LceBconv2dOp $input, $filter,
            (TFL_DequantizeOp (TFL_QuantizeOp $post_activation_multiplier, $qtype1)),
            (TFL_DequantizeOp (TFL_QuantizeOp $post_activation_bias, $qtype2)),
            $channels_in, $strides, $padding, $pad_values, $data_format,
            $dilations, $filter_format, $act),
          (TF_LceBconv2dOp $input, $filter,
            $post_activation_multiplier, $post_activation_bias,
            $channels_in, $strides, $padding, $pad_values, $data_format,
            $dilations, $filter_format, $act)>;
