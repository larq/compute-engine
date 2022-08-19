include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

class ConstantValue<string val> : AttrConstraint<CPred<"IsConstantValue($_self, " # val # ")">>;

def : Pat<(LQ_QuantizeOp
              (TFL_GreaterEqualOp:$ge_op
                  $input,
                  (Arith_ConstantOp ConstantValue<"0.0f">))),
          (LQ_QuantizeOp $input),
          [(HasOneUse $ge_op)],
          (addBenefit 150)>;

def : Pat<(LQ_QuantizeOp
              (TFL_GreaterEqualOp:$ge_op
                  $input,
                  $threshold)),
          (LQ_QuantizeOp
              (TFL_SubOp $input, $threshold, TFL_AF_None)),
          [(HasOneUse $ge_op)],
          (addBenefit 100)>;

def : Pat<(LQ_QuantizeOp
              (TFL_LessEqualOp:$ge_op $lhs, $rhs)),
          (LQ_QuantizeOp
              (TFL_GreaterEqualOp $rhs, $lhs)),
          [(HasOneUse $ge_op)],
          (addBenefit 100)>;

// TODO: Check shapes before fusing
multiclass FuseAddOrSubWithBConv2D<Op binaryOp> {
  def : Pat<(binaryOp
                (LQ_Bconv2dOp:$output
                    $input,
                    $filter,
                    $post_activation_multiplier,
                    (Arith_ConstantOp F32ElementsAttr:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    $fused_activation_function,
                    $pad_values,
                    $padding,
                    $stride_height,
                    $stride_width),
                (Arith_ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (LQ_Bconv2dOp
                $input,
                $filter,
                $post_activation_multiplier,
                (binaryOp (Arith_ConstantOp $post_activation_bias),
                          (Arith_ConstantOp $value), TFL_AF_None),
                $output_threshold,
                $channels_in,
                $dilation_height_factor,
                $dilation_width_factor,
                $fused_activation_function,
                $pad_values,
                $padding,
                $stride_height,
                $stride_width),
            [(HasOneUse $output)], (addBenefit 100)>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseAddOrSubWithBConv2D<binaryOp>;

// TODO: Check shapes before fusing
multiclass FuseMulOrDivWithBConv2D<Op binaryOp> {
  def : Pat<(binaryOp
                (LQ_Bconv2dOp:$conv_output
                    $input,
                    $filter,
                    (Arith_ConstantOp F32ElementsAttr:$post_activation_multiplier),
                    (Arith_ConstantOp F32ElementsAttr:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    $fused_activation_function,
                    $pad_values,
                    $padding,
                    $stride_height,
                    $stride_width),
                (Arith_ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (binaryOp (Arith_ConstantOp $post_activation_multiplier),
                          (Arith_ConstantOp $value), TFL_AF_None),
                (binaryOp (Arith_ConstantOp $post_activation_bias),
                          (Arith_ConstantOp $value), TFL_AF_None),
                $output_threshold,
                $channels_in,
                $dilation_height_factor,
                $dilation_width_factor,
                $fused_activation_function,
                $pad_values,
                $padding,
                $stride_height,
                $stride_width),
            [(HasOneUse $conv_output)], (addBenefit 100)>;
}
foreach binaryOp = [TFL_DivOp, TFL_MulOp] in
  defm : FuseMulOrDivWithBConv2D<binaryOp>;


// Fuse an activation function into the BConv2D.
multiclass FuseActFnIntoConvOpPat<Op ActFnOp, ConstantStrAttr ActFnAttr> {
  def : Pat<(ActFnOp
                (LQ_Bconv2dOp:$conv_output
                    $input,
                    $filter,
                    (Arith_ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                    (Arith_ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    TFL_AF_None,
                    $pad_values,
                    TFL_PAD_Valid:$padding,
                    $stride_height,
                    $stride_width)),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (Arith_ConstantOp $post_activation_multiplier),
                (Arith_ConstantOp $post_activation_bias),
                $output_threshold,
                $channels_in,
                $dilation_height_factor,
                $dilation_width_factor,
                ActFnAttr,
                $pad_values,
                $padding,
                $stride_height,
                $stride_width),
            [(HasOneUse $conv_output)], (addBenefit 100)>;
  def : Pat<(ActFnOp
                (LQ_Bconv2dOp:$conv_output
                    $input,
                    $filter,
                    (Arith_ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                    (Arith_ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    TFL_AF_None,
                    ConstantAttr<I32Attr, "1">:$pad_values,
                    TFL_PAD_Same:$padding,
                    $stride_height,
                    $stride_width)),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (Arith_ConstantOp $post_activation_multiplier),
                (Arith_ConstantOp $post_activation_bias),
                $output_threshold,
                $channels_in,
                $dilation_height_factor,
                $dilation_width_factor,
                ActFnAttr,
                $pad_values,
                $padding,
                $stride_height,
                $stride_width),
            [(HasOneUse $conv_output)], (addBenefit 100)>;
}
foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu1Op, TFL_AF_Relu1],
                     [TFL_Relu6Op, TFL_AF_Relu6]] in
  defm : FuseActFnIntoConvOpPat<!cast<Op>(actFnPair[0]), !cast<ConstantStrAttr>(actFnPair[1])>;
