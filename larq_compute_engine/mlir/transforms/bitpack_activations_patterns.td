include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def CreateNoneAttrValue : NativeCodeCall<"$_builder.getUnitAttr()">;
def GetSignsOfVectorAndBroadcast4D : NativeCodeCall<"GetSignsOfVectorAndBroadcast4D($0)">;
def GetBitpackedOutputThresholds : NativeCodeCall<"GetBitpackedOutputThresholds($_builder, $0, $1, $2, $3)">;

class WriteBitpackedActivationsPat<StrEnumAttrCase padding_type, string pad_values> :
      Pat<(LQ_QuantizeOp
              (LQ_Bconv2dOp:$output
                  $input,
                  (Arith_ConstantOp F32ElementsAttr:$filter),
                  (Arith_ConstantOp F32ElementsAttr:$post_activation_multiplier),
                  (Arith_ConstantOp F32ElementsAttr:$post_activation_bias),
                  (ConstantOp UnitAttr),
                  $channels_in,
                  $dilation_height_factor,
                  $dilation_width_factor,
                  $fused_activation_function,
                  ConstantAttr<I32Attr, pad_values>,
                  padding_type,
                  $stride_height,
                  $stride_width)),
          (LQ_Bconv2dOp
              $input,
              (TFL_MulOp
                  (Arith_ConstantOp $filter),
                  (Arith_ConstantOp
                      (GetSignsOfVectorAndBroadcast4D $post_activation_multiplier)),
                  TFL_AF_None),
              (ConstantOp (CreateNoneAttrValue)),
              (ConstantOp (CreateNoneAttrValue)),
              (Arith_ConstantOp
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
              padding_type,
              $stride_height,
              $stride_width),
          [(HasOneUse $output)], (addBenefit 10)>;
def : WriteBitpackedActivationsPat<TFL_PAD_Valid, "0">;
def : WriteBitpackedActivationsPat<TFL_PAD_Same, "1">;
