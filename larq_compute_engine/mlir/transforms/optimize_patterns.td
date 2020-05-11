include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

// TODO: Check shapes before fusing
multiclass FuseAddOrSubWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LceBconv2dOp:$output $input, $filter,
                          $post_activation_multiplier, (ConstantOp F32ElementsAttr:$post_activation_bias),
                          $channels_in, $strides, $padding, $pad_values,
                          $data_format, $dilations, $filter_format,
                          $read_bitpacked_input, $write_bitpacked_output, $activation),
                      (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (TF_LceBconv2dOp $input, $filter, $post_activation_multiplier,
                          (binaryOp (ConstantOp $post_activation_bias), (ConstantOp $value), TFL_AF_None),
                          $channels_in, $strides, $padding, $pad_values, $data_format,
                          $dilations, $filter_format, $read_bitpacked_input,
                          $write_bitpacked_output, $activation),
            [(HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseAddOrSubWithBConv2D<binaryOp>;

// TODO: Check shapes before fusing
multiclass FuseMulOrDivWithBConv2D<dag binaryOp> {
  def : Pat<(binaryOp (TF_LceBconv2dOp:$conv_output $input, $filter,
                        (ConstantOp F32ElementsAttr:$post_activation_multiplier),
                        (ConstantOp F32ElementsAttr:$post_activation_bias),
                        $channels_in, $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $read_bitpacked_input,
                        $write_bitpacked_output, $activation),
                      (ConstantOp F32ElementsAttr:$value), TFL_AF_None),
            (TF_LceBconv2dOp $input, $filter,
                        (binaryOp (ConstantOp $post_activation_multiplier),
                                  (ConstantOp $value), TFL_AF_None),
                        (binaryOp (ConstantOp $post_activation_bias),
                                  (ConstantOp $value), TFL_AF_None),
                        $channels_in, $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $read_bitpacked_input,
                        $write_bitpacked_output, $activation),
         [(HasOneUse $conv_output)]>;
}

foreach binaryOp = [TFL_DivOp, TFL_MulOp] in
  defm : FuseMulOrDivWithBConv2D<binaryOp>;

class ConstantValue<string val> : AttrConstraint<CPred<"IsConstantValue($_self, " # val # ")">>;

// Fuse activation function into BConv2D
multiclass FuseActFnIntoConvOpPat<dag ActFnOp, dag ActFnAttr> {
  def : Pat<(ActFnOp (TF_LceBconv2dOp:$conv_output $input, $filter,
                        (ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                        (ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                        $channels_in, $strides, ConstantAttr<StrAttr, "VALID">:$padding,
                        $pad_values, $data_format, $dilations, $filter_format,
                        $read_bitpacked_input, $write_bitpacked_output,
                        TFL_AF_None)),
            (TF_LceBconv2dOp $input, $filter,
                        (ConstantOp $post_activation_multiplier),
                        (ConstantOp $post_activation_bias),
                        $channels_in, $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $read_bitpacked_input,
                        $write_bitpacked_output, ActFnAttr),
         [(HasOneUse $conv_output)]>;
  def : Pat<(ActFnOp (TF_LceBconv2dOp:$conv_output $input, $filter,
                        (ConstantOp ConstantValue<"1.0f">:$post_activation_multiplier),
                        (ConstantOp ConstantValue<"0.0f">:$post_activation_bias),
                        $channels_in,
                        $strides,
                        ConstantAttr<StrAttr, "SAME">:$padding,
                        ConstantAttr<I32Attr, "1">:$pad_values,
                        $data_format, $dilations, $filter_format,
                        $read_bitpacked_input, $write_bitpacked_output,
                        TFL_AF_None)),
            (TF_LceBconv2dOp $input, $filter,
                        (ConstantOp $post_activation_multiplier),
                        (ConstantOp $post_activation_bias),
                        $channels_in, $strides, $padding, $pad_values, $data_format,
                        $dilations, $filter_format, $read_bitpacked_input,
                        $write_bitpacked_output, ActFnAttr),
         [(HasOneUse $conv_output)]>;
}

foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu1Op, TFL_AF_Relu1],
                     [TFL_Relu6Op, TFL_AF_Relu6]] in
  defm : FuseActFnIntoConvOpPat<actFnPair[0], actFnPair[1]>;


// Bitpack weights
def Conv2DFilter : AttrConstraint<CPred<"IsConv2DFilter($_self)">>;
def Bitpack: NativeCodeCall<"Bitpack($_builder, $0)">;

def : Pat<(TF_LceBconv2dOp $input, (ConstantOp Conv2DFilter:$filter),
            $post_activation_multiplier, $post_activation_bias,
            $channels_in, $strides, $padding, $pad_values, $data_format, $dilations,
            ConstantAttr<StrAttr, "OHWI">, $read_bitpacked_input,
            $write_bitpacked_output, $act),
          (TF_LceBconv2dOp $input, (ConstantOp (Bitpack $filter)),
            $post_activation_multiplier, $post_activation_bias,
            $channels_in, $strides, $padding, $pad_values, $data_format, $dilations,
            ConstantAttr<StrAttr, "OHWI_PACKED">, $read_bitpacked_input,
            $write_bitpacked_output, $act)>;
