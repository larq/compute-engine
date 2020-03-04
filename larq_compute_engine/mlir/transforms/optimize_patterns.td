include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0->hasOneUse()">>;

// TODO: Check shapes before fusing
multiclass FuseAddOrSubWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LqceBconv2d64Op:$output $input, $filter,
                          $post_activation_multiplier, (ConstantOp F32ElementsAttr:$post_activation_bias), $strides,
                          $padding, $pad_values, $data_format, $dilations, $filter_format, $activation),
                      (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (TF_LqceBconv2d64Op $input, $filter, $post_activation_multiplier,
                          (binaryOp (ConstantOp $post_activation_bias), (ConstantOp $value), TFL_AF_None),
                          $strides, $padding, $pad_values, $data_format,
                          $dilations, $filter_format, $activation),
            [(HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseAddOrSubWithBConv2D<binaryOp>;

// TODO: Check shapes before fusing
multiclass FuseMulOrDivWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LqceBconv2d64Op:$conv_output $input, $filter,
                        (ConstantOp F32ElementsAttr:$post_activation_multiplier),
                        (ConstantOp F32ElementsAttr:$post_activation_bias),
                        $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $activation),
                      (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (TF_LqceBconv2d64Op $input, $filter,
                        (binaryOp (ConstantOp $post_activation_multiplier),
                                  (ConstantOp $value), TFL_AF_None),
                        (binaryOp (ConstantOp $post_activation_bias),
                                  (ConstantOp $value), TFL_AF_None),
                        $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $activation),
         [(HasOneUse $conv_output)]>;
}

foreach binaryOp = [TFL_DivOp, TFL_MulOp] in
  defm : FuseMulOrDivWithBConv2D<binaryOp>;

class IsConstantValue<string val> : Constraint<CPred<"IsConstantValue($0, " # val # ")">>;

// Fuse activation function into BConv2D
multiclass FuseActFnIntoConvOpPat<dag ActFnOp, dag ActFnAttr> {
  def : Pat<(ActFnOp (TF_LqceBconv2d64Op:$conv_output $input, $filter,
                        (ConstantOp $post_activation_multiplier),
                        (ConstantOp $post_activation_bias),
                        $strides,
                        ConstantAttr<StrAttr, "VALID">: $padding,
                        $pad_values,
                        $data_format,
                        $dilations, $filter_format, TFL_AF_None)),
            (TF_LqceBconv2d64Op $input, $filter,
                        (ConstantOp $post_activation_multiplier), (ConstantOp $post_activation_bias),
                        $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, ActFnAttr),
         [(HasOneUse $conv_output),
         (IsConstantValue<"1.0f"> $post_activation_multiplier),
         (IsConstantValue<"0.0f"> $post_activation_bias)]>;
  def : Pat<(ActFnOp (TF_LqceBconv2d64Op:$conv_output $input, $filter,
                        (ConstantOp $post_activation_multiplier),
                        (ConstantOp $post_activation_bias),
                        $strides,
                        ConstantAttr<StrAttr, "SAME">: $padding,
                        ConstantAttr<I32Attr, "1">: $pad_values,
                        $data_format,
                        $dilations, $filter_format, TFL_AF_None)),
            (TF_LqceBconv2d64Op $input, $filter,
                        (ConstantOp $post_activation_multiplier), (ConstantOp $post_activation_bias),
                        $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, ActFnAttr),
         [(HasOneUse $conv_output),
          (IsConstantValue<"1.0f"> $post_activation_multiplier),
          (IsConstantValue<"0.0f"> $post_activation_bias)]>;
}

foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu6Op, TFL_AF_Relu6]] in
  defm : FuseActFnIntoConvOpPat<actFnPair[0], actFnPair[1]>;
