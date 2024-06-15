include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getShapedType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def CreateNoneValue : NativeCodeCall<
  "$_builder.create<TFL::NoValueOp>($0.getLoc(), $_builder.getUnitAttr())">;
def GetSignsOfVectorAndBroadcast4D : NativeCodeCall<"GetSignsOfVectorAndBroadcast4D($0)">;
def GetBitpackedOutputThresholds : NativeCodeCall<"GetBitpackedOutputThresholds($_builder, $0, $1, $2, $3)">;

class WriteBitpackedActivationsPat<ConstantStrAttr padding_type, string pad_values> :
      Pat<(LQ_QuantizeOp
              (LQ_Bconv2dOp:$output
                  $input,
                  (Arith_ConstantOp F32ElementsAttr:$filter),
                  (Arith_ConstantOp F32ElementsAttr:$post_activation_multiplier),
                  (Arith_ConstantOp F32ElementsAttr:$post_activation_bias),
                  (TFL_NoValueOp UnitAttr),
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
              (CreateNoneValue $input),
              (CreateNoneValue $input),
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
          [(HasOneUse $output)]>;
def : WriteBitpackedActivationsPat<TFL_PAD_Valid, "0">;
def : WriteBitpackedActivationsPat<TFL_PAD_Same, "1">;
