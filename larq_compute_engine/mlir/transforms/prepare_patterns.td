include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/lite/transforms/prepare_patterns.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


// This relies on implementation details of larq.math.sign. We should make
// this more general in the future
def : Pat<(TF_SignOp (TF_AddV2Op (TF_SignOp $arg), $c)), (TF_LqceBsignOp $arg),
          [], (addBenefit 100)>;
def : Pat<(TF_SignOp (TF_AddV2Op $c, (TF_SignOp $arg))), (TF_LqceBsignOp $arg),
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

def GetBias : NativeCodeCall<"GetBias($0)">;
def GetMultiplier : NativeCodeCall<"GetMultiplier($0)">;
def BinaryFilter : Constraint<CPred<"IsBinaryFilter($0)">>;

def : Pat<(TF_Conv2DOp (TF_LqceBsignOp $input), $filter, $strides, $use_cudnn,
                       $padding, $explicit_padding, $data_format, $dilations),
          (TF_LqceBconv2d64Op $input,
            (TF_TransposeOp $filter, (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{3, 0, 1, 2}">)),
            (TF_ConstOp (GetMultiplier $filter)),
            (TF_ConstOp (GetBias $filter)),
            $strides, $padding, $explicit_padding,
            $data_format, $dilations, ConstantAttr<StrAttr, "OHWI">),
          [(BinaryFilter $filter)], (addBenefit 90)>;
