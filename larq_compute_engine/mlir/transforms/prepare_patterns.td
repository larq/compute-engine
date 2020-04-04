include "tensorflow/compiler/mlir/lite/transforms/prepare_patterns.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


// This relies on implementation details of larq.math.sign. We should make
// this more general in the future
def : Pat<(TF_SignOp (TF_AddV2Op (TF_SignOp $arg), $c)), (TF_LceBsignOp $arg),
          [], (addBenefit 100)>;
def : Pat<(TF_SignOp (TF_AddV2Op $c, (TF_SignOp $arg))), (TF_LceBsignOp $arg),
          [], (addBenefit 100)>;

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

class GetConstantVector<string val> : NativeCodeCall<"GetConstantVector($0, " # val # ")">;
def BinaryFilter : Constraint<CPred<"IsBinaryFilter($0)">>;
def GetScaleVector : NativeCodeCall<"GetScaleVector($0)">;

def : Pat<(TF_Conv2DOp (TF_LceBsignOp $input), (ConstantOp $filter), $strides, $use_cudnn,
                       $padding, $explicit_padding, IsDataFormatNHWC: $data_format, $dilations),
          (TF_LceBconv2dOp $input,
            (TF_TransposeOp (TF_DivOp (ConstantOp $filter), (ConstantOp (GetScaleVector $filter))), (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
            (ConstantOp (GetScaleVector $filter)),
            (ConstantOp (GetConstantVector<"0.0f"> $filter)),
            $strides, $padding, ConstantAttr<I32Attr, "0">, $data_format,
            $dilations, ConstantAttr<StrAttr, "OHWI">, TFL_AF_None),
          [(BinaryFilter $filter)], (addBenefit 90)>;

def ConstFloatValueIsOne : Constraint<
  CPred<"$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == 1.0f">>;

def SamePadding : Constraint<CPred<"IsSamePadding($0, $1, $2, $3)">>;

def : Pat<(TF_Conv2DOp: $output
            (TF_PadV2Op (TF_LceBsignOp $input), (ConstantOp $paddings), (ConstantOp $pad_values)),
            (ConstantOp $filter), $strides, $use_cudnn, ConstantAttr<StrAttr, "VALID">,
            $explicit_padding, IsDataFormatNHWC: $data_format, $dilations),
          (TF_LceBconv2dOp $input,
            (TF_TransposeOp (TF_DivOp (ConstantOp $filter), (ConstantOp (GetScaleVector $filter))), (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
            (ConstantOp (GetScaleVector $filter)),
            (ConstantOp (GetConstantVector<"0.0f"> $filter)),
            $strides,
            ConstantAttr<StrAttr, "SAME">,
            ConstantAttr<I32Attr, "1">,
            $data_format,
            $dilations,
            ConstantAttr<StrAttr, "OHWI">,
            TFL_AF_None),
          [(BinaryFilter $filter),
           (ConstFloatValueIsOne $pad_values),
           (SamePadding $paddings, $input, $output, $strides)],
          (addBenefit 90)>;
