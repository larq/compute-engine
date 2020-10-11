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

// Match a binary convolution with bipolar (+1/-1) inputs, padded (if
// applicable) with zeroes.
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

// Match a binary convolution with bipolar (+1/-1) inputs, padded with ones.
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

// Match a binary convolution with unipolar (+1/0) inputs, padded (if
// applicable) with zeroes. The weights are bipolar (+1/-1). We use the
// following trick to perform this computation using the bconv2d op (which only
// supports bipolar inputs).
//     Suppose `a` is a unipolar (+1/0) activation and `w` is a bipolar (+1/-1)
// weight. If we define `b` to be 1 if `a` is 1 and `-1` if `a` is 0, then
// `a = 0.5 * b + 0.5`. This means that `a * w` can be computed with bipolar
// multiplication followed by a non-binary multiplication and bias:
//            `a * w = (0.5 * b + 0.5) * w = 0.5 * b * w + 0.5 * w`
// Note that the bias `0.5 * w` is a constant at conversion-time. The
// generalisation to more than a single activation/weight element is then:
//         `dotprod(a, w) = 0.5 * dotprod(b, w) + 0.5 * reduce_sum(w)`
// which is the form we use below.
//     There is one final thing to note here. Because the unipolar input 1 is
// represented by bipolar input 1 and unipolar input 0 by bipolar input -1, TF
// zero-padding is equivalent to padding with -1 in LCE; however, the bconv2d op
// only supports padding with 1. The solution is to multiply the (bipolar) input
// by -1 and the weights by -1 -- this means that TF zero-padding is equivalent
// to LCE bconv2d one-padding.
def : Pat<(TF_Conv2DOp
              (TF_AddV2Op
                  (TF_MulOp
                      (LQ_DequantizeOp (LQ_QuantizeOp $unquantized_input)),
                      (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
                  (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
              (ConstantOp $filter),
              IsIntList1XY1:$strides,
              $use_cudnn,
              $padding,
              $explicit_padding,
              IsDataFormatNHWC: $data_format,
              IsIntList1XY1:$dilations),
          (LQ_Bconv2dOp
              (LQ_QuantizeOp
                  (TF_MulOp
                      $unquantized_input,
                      (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "-1.0f">))),
              (TF_TransposeOp
                  (TF_MulOp
                      (TF_DivOp
                          (ConstantOp $filter),
                          (ConstantOp (GetScaleVector $filter))),
                      (ConstantOp (GetConstantVector<"-1.0f"> $filter))),
                  (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
              (TF_MulOp
                  (ConstantOp (GetScaleVector $filter)),
                  (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
              (TF_MulOp
                  (TF_SumOp
                      (TF_DivOp
                          (ConstantOp $filter),
                          (ConstantOp (GetScaleVector $filter))),
                      (ConstantOp ConstantAttr<I32VectorElementsAttr<3>, "{0, 1, 2}">),
                      ConstBoolAttrFalse),
                  (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
              (ConstantOp (CreateNoneAttrValue)),
              (GetNumChannels $unquantized_input),
              ExtractI32At<1>:$dilations,
              ExtractI32At<2>:$dilations,
              TFL_AF_None,
              ConstantAttr<I32Attr, "1">,
              $padding,
              ExtractI32At<1>:$strides,
              ExtractI32At<2>:$strides),
          [(BinaryFilter $filter)],
          (addBenefit 90)>;
