include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"


def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

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
