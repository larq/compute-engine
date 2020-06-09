include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"
include "larq_compute_engine/mlir/transforms/op_removal_patterns.td"

// Bitpack weights
def Conv2DFilter : AttrConstraint<CPred<"IsConv2DFilter($_self)">>;
def Bitpack : NativeCodeCall<"Bitpack($_builder, $0)">;

def : Pat<(LQ_Bconv2dOp
              $input,
              (ConstantOp Conv2DFilter:$filter),
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
              $stride_width),
          (LQ_Bconv2dOp
              $input,
              (ConstantOp (Bitpack $filter)),
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
              $stride_width),
          []>;
