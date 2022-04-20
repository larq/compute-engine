include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"
include "larq_compute_engine/mlir/transforms/op_removal_patterns.td"

class ConstantValue<string val> : AttrConstraint<CPred<"IsConstantValue($_self, " # val # ")">>;

def CreateTFBroadcastToOp : NativeCodeCall<
    "$_builder.create<TF::BroadcastToOp>("
        "$0.getLoc(),"
        "RankedTensorType::get("
            "$0.getType().cast<RankedTensorType>().getShape(),"
            "getElementTypeOrSelf($1.getType())),"
        "$1,"
        "$2)">;

def CreateTFShapeOp : NativeCodeCall<
    "$_builder.create<TF::ShapeOp>($0.getLoc(), $1, $2)">;

// Base quantiser patterns that match the `tf.where` implementation of `ste_sign`.
multiclass QuantDequantPatterns<Op SelectOp> {
  def : Pat<(SelectOp:$select_op
                $cond,
                (Arith_ConstantOp ConstantValue<"1.0f">),
                (Arith_ConstantOp ConstantValue<"-1.0f">)),
            (LQ_DequantizeOp
                (LQ_QuantizeOp
                    (CreateTFBroadcastToOp
                        $select_op,
                        $cond,
                        (CreateTFShapeOp
                            $select_op,
                            $select_op,
                            /*use 32bit*/ConstBoolAttrFalse)))),
            [], (addBenefit 100)>;
  def : Pat<(SelectOp:$select_op
                $cond,
                (Arith_ConstantOp ConstantValue<"-1.0f">),
                (Arith_ConstantOp ConstantValue<"1.0f">)),
            (LQ_DequantizeOp
                (LQ_QuantizeOp
                    (CreateTFBroadcastToOp
                        $select_op,
                        (TF_LogicalNotOp $cond),
                        (CreateTFShapeOp
                            $select_op,
                            $select_op,
                            /*use 32bit*/ConstBoolAttrFalse)))),
            [], (addBenefit 100)>;
}
foreach SelectOp = [TF_SelectOp, TF_SelectV2Op]<Op> in
  defm : QuantDequantPatterns<SelectOp>;

// A fallback for the old version of `ste_sign` that uses a specific `tf.sign`
// based implementation of `larq.math.sign`.
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
def ValidFilterShape : Constraint<CPred<"HasValidFilterShape($0, $1)">>;
def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">;
def CreateNoneAttrValue : NativeCodeCall<"$_builder.getUnitAttr()">;

// All targets support this pattern with "VALID" padding, but only the "arm"
// target supports it with "SAME" padding.
class PrepareBConvPadValue0Pat<StrEnumAttrCase padding_type> :
      Pat<(TF_Conv2DOp
              (LQ_DequantizeOp:$dequantized_input $input),
              (Arith_ConstantOp:$filter_op $filter),
              IsIntList1XY1:$strides,
              $use_cudnn,
              padding_type:$padding,
              $explicit_padding,
              IsDataFormatNHWC:$data_format,
              IsIntList1XY1:$dilations),
          (LQ_Bconv2dOp
              $input,
              (TF_TransposeOp
                  (TF_DivOp
                      (Arith_ConstantOp $filter),
                      (Arith_ConstantOp (GetScaleVector $filter))),
                  (Arith_ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
              (Arith_ConstantOp (GetScaleVector $filter)),
              (Arith_ConstantOp (GetConstantVector<"0.0f"> $filter)),
              (ConstantOp (CreateNoneAttrValue)),
              (GetNumChannels $dequantized_input),
              ExtractI32At<1>:$dilations,
              ExtractI32At<2>:$dilations,
              TFL_AF_None,
              ConstantAttr<I32Attr, "0">,
              $padding,
              ExtractI32At<1>:$strides,
              ExtractI32At<2>:$strides),
          [(BinaryFilter $filter),
           (ValidFilterShape $dequantized_input, $filter_op)],
          (addBenefit 90)>;
def : PrepareBConvPadValue0Pat<TFL_PAD_Valid>;

def ConstFloatValueIsOne : Constraint<
  CPred<"$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == 1.0f">>;

def SamePadding : Constraint<CPred<"IsSamePadding($0, $1, $2, $3)">>;

def : Pat<(TF_Conv2DOp:$output
              (TF_PadV2Op
                  (LQ_DequantizeOp:$dequantized_input $input),
                  (Arith_ConstantOp $paddings),
                  (Arith_ConstantOp $pad_values)),
              (Arith_ConstantOp:$filter_op $filter),
              IsIntList1XY1:$strides,
              $use_cudnn,
              TFL_PAD_Valid,
              $explicit_padding,
              IsDataFormatNHWC:$data_format,
              IsIntList1XY1:$dilations),
          (LQ_Bconv2dOp $input,
              (TF_TransposeOp
                  (TF_DivOp
                      (Arith_ConstantOp $filter),
                      (Arith_ConstantOp (GetScaleVector $filter))),
                  (Arith_ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
              (Arith_ConstantOp (GetScaleVector $filter)),
              (Arith_ConstantOp (GetConstantVector<"0.0f"> $filter)),
              (ConstantOp (CreateNoneAttrValue)),
              (GetNumChannels $dequantized_input),
              ExtractI32At<1>:$dilations,
              ExtractI32At<2>:$dilations,
              TFL_AF_None,
              ConstantAttr<I32Attr, "1">,
              TFL_PAD_Same,
              ExtractI32At<1>:$strides,
              ExtractI32At<2>:$strides),
          [(BinaryFilter $filter),
           (ConstFloatValueIsOne $pad_values),
           (SamePadding $paddings, $input, $output, $strides),
           (ValidFilterShape $dequantized_input, $filter_op)],
          (addBenefit 90)>;
