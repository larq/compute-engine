include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"
include "larq_compute_engine/mlir/transforms/op_removal_patterns.td"


// This relies on implementation details of larq.math.sign. We should make
// this more general in the future
def : Pat<(TF_SignOp (TF_AddV2Op (TF_SignOp $arg), $c)),
          (LQ_DequantizeOp (LQ_QuantizeOp $arg)), [], (addBenefit 100)>;
def : Pat<(TF_SignOp (TF_AddV2Op $c, (TF_SignOp $arg))),
          (LQ_DequantizeOp (LQ_QuantizeOp $arg)), [], (addBenefit 100)>;

// Copied from legalize_patterns.td
class I32VectorElementsAttr<int len> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isInteger(32)">,
  "32-bit int elements attribute of shape [" # len # "]"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({" # len # "}, $_builder.getIntegerType(32)), $0)";
}

// Extract the ith int element from an ArrayAttr $0 as an 32-bit IntegerAttr
// with builder.
class ExtractI32At<int i> : NativeCodeCall<
    "$_builder.getI32IntegerAttr($_self.cast<ArrayAttr>().getValue()[" # i #
    "].cast<IntegerAttr>().getInt())">;
def IsIntList1XY1 : AttrConstraint<CPred<"TFIntListIs1XY1($_self)">>;

class GetConstantVector<string val> : NativeCodeCall<"GetConstantVector($0, " # val # ")">;
def BinaryFilter : Constraint<CPred<"IsBinaryFilter($0)">>;
def GetScaleVector : NativeCodeCall<"GetScaleVector($0)">;
def GetNumChannels : NativeCodeCall<"GetNumChannels($_builder, $0)">;
def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">;
def CreateNoneAttrValue : NativeCodeCall<"$_builder.getUnitAttr()">;

def : Pat<(TF_Conv2DOp
              (LQ_DequantizeOp: $dequantized_input $input),
              (ConstantOp $filter),
              IsIntList1XY1:$strides,
              $use_cudnn,
              $padding,
              $explicit_padding,
              IsDataFormatNHWC: $data_format,
              IsIntList1XY1:$dilations),
          (LQ_Bconv2dOp
              $input,
              (TF_TransposeOp
                  (TF_DivOp
                      (ConstantOp $filter),
                      (ConstantOp (GetScaleVector $filter))),
                  (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
              (ConstantOp (GetScaleVector $filter)),
              (ConstantOp (GetConstantVector<"0.0f"> $filter)),
              (ConstantOp (CreateNoneAttrValue)),
              (GetNumChannels $dequantized_input),
              ExtractI32At<1>:$dilations,
              ExtractI32At<2>:$dilations,
              TFL_AF_None,
              ConstantAttr<I32Attr, "0">,
              $padding,
              ExtractI32At<1>:$strides,
              ExtractI32At<2>:$strides),
          [(BinaryFilter $filter)],
          (addBenefit 90)>;

def ConstFloatValueIsOne : Constraint<
  CPred<"$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == 1.0f">>;

def SamePadding : Constraint<CPred<"IsSamePadding($0, $1, $2, $3)">>;

def : Pat<(TF_Conv2DOp:$output
              (TF_PadV2Op
                  (LQ_DequantizeOp: $dequantized_input $input),
                  (ConstantOp $paddings),
                  (ConstantOp $pad_values)),
              (ConstantOp $filter),
              IsIntList1XY1:$strides,
              $use_cudnn,
              ConstantAttr<StrAttr, "VALID">,
              $explicit_padding,
              IsDataFormatNHWC:$data_format,
              IsIntList1XY1:$dilations),
          (LQ_Bconv2dOp $input,
              (TF_TransposeOp
                  (TF_DivOp
                      (ConstantOp $filter),
                      (ConstantOp (GetScaleVector $filter))),
                  (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
              (ConstantOp (GetScaleVector $filter)),
              (ConstantOp (GetConstantVector<"0.0f"> $filter)),
              (ConstantOp (CreateNoneAttrValue)),
              (GetNumChannels $dequantized_input),
              ExtractI32At<1>:$dilations,
              ExtractI32At<2>:$dilations,
              TFL_AF_None,
              ConstantAttr<I32Attr, "1">,
              ConstantAttr<StrAttr, "SAME">,
              ExtractI32At<1>:$strides,
              ExtractI32At<2>:$strides),
          [(BinaryFilter $filter),
           (ConstFloatValueIsOne $pad_values),
           (SamePadding $paddings, $input, $output, $strides)],
          (addBenefit 90)>;
