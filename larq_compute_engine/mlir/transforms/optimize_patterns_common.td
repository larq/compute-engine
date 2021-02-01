include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

class ConstantValue<string val> : AttrConstraint<CPred<"IsConstantValue($_self, " # val # ")">>;

// TODO: Check shapes before fusing
multiclass FuseAddOrSubWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp
                (LQ_Bconv2dOp:$output
                    $input,
                    $filter,
                    $post_activation_multiplier,
                    (ConstantOp F32ElementsAttr:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    $fused_activation_function,
                    $pad_values,
                    $padding,
                    $stride_height,
                    $stride_width),
                (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (LQ_Bconv2dOp
                $input,
                $filter,
                $post_activation_multiplier,
                (binaryOp (ConstantOp $post_activation_bias),
                          (ConstantOp $value), TFL_AF_None),
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
multiclass FuseMulOrDivWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp
                (LQ_Bconv2dOp:$conv_output
                    $input,
                    $filter,
                    (ConstantOp F32ElementsAttr:$post_activation_multiplier),
                    (ConstantOp F32ElementsAttr:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    $fused_activation_function,
                    $pad_values,
                    $padding,
                    $stride_height,
                    $stride_width),
                (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (binaryOp (ConstantOp $post_activation_multiplier),
                          (ConstantOp $value), TFL_AF_None),
                (binaryOp (ConstantOp $post_activation_bias),
                          (ConstantOp $value), TFL_AF_None),
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
multiclass FuseActFnIntoConvOpPat<dag ActFnOp, dag ActFnAttr> {
  def : Pat<(ActFnOp
                (LQ_Bconv2dOp:$conv_output
                    $input,
                    $filter,
                    (ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                    (ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    TFL_AF_None,
                    $pad_values,
                    ConstantAttr<StrAttr, "VALID">:$padding,
                    $stride_height,
                    $stride_width)),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (ConstantOp $post_activation_multiplier),
                (ConstantOp $post_activation_bias),
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
                    (ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                    (ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                    $output_threshold,
                    $channels_in,
                    $dilation_height_factor,
                    $dilation_width_factor,
                    TFL_AF_None,
                    ConstantAttr<I32Attr, "1">:$pad_values,
                    ConstantAttr<StrAttr, "SAME">:$padding,
                    $stride_height,
                    $stride_width)),
            (LQ_Bconv2dOp
                $input,
                $filter,
                (ConstantOp $post_activation_multiplier),
                (ConstantOp $post_activation_bias),
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
  defm : FuseActFnIntoConvOpPat<actFnPair[0], actFnPair[1]>;

def CreateNoneAttrValue : NativeCodeCall<"$_builder.getUnitAttr()">;
def GetSignsOfVectorAndBroadcast4D : NativeCodeCall<"GetSignsOfVectorAndBroadcast4D($0)">;
def GetBitpackedOutputThresholds : NativeCodeCall<"GetBitpackedOutputThresholds($_builder, $0, $1, $2, $3)">;

class WriteBitpackedActivationsPat<string padding_type, string pad_values> :
      Pat<(LQ_QuantizeOp
              (LQ_Bconv2dOp:$output
                  $input,
                  (ConstantOp F32ElementsAttr:$filter),
                  (ConstantOp F32ElementsAttr:$post_activation_multiplier),
                  (ConstantOp F32ElementsAttr:$post_activation_bias),
                  (ConstantOp UnitAttr),
                  $channels_in,
                  $dilation_height_factor,
                  $dilation_width_factor,
                  $fused_activation_function,
                  ConstantAttr<I32Attr, pad_values>,
                  ConstantAttr<StrAttr, padding_type>,
                  $stride_height,
                  $stride_width)),
          (LQ_Bconv2dOp
              $input,
              (TFL_MulOp
                  (ConstantOp $filter),
                  (ConstantOp
                      (GetSignsOfVectorAndBroadcast4D $post_activation_multiplier)),
                  TFL_AF_None),
              (ConstantOp (CreateNoneAttrValue)),
              (ConstantOp (CreateNoneAttrValue)),
              (ConstantOp
                  (GetBitpackedOutputThresholds
                      $filter,
                      $post_activation_multiplier,
                      $post_activation_bias,
                      $fused_activation_function)),
              $channels_in,
              $dilation_height_factor,
              $dilation_width_factor,
              $fused_activation_function,
              ConstantAttr<I32Attr, pad_values>,
              ConstantAttr<StrAttr, padding_type>,
              $stride_height,
              $stride_width),
          [(HasOneUse $output)], (addBenefit 10)>;
def : WriteBitpackedActivationsPat<"VALID", "0">;
def : WriteBitpackedActivationsPat<"SAME", "1">;
