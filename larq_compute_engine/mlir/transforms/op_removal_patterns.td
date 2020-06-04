include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

def NonOpaqueElementsAttr : ElementsAttrBase<
  CPred<"!$_self.isa<OpaqueElementsAttr>()">,
  "non-opaque constant tensor">;

// Convert to std constant for statically shaped, non-opaque constants.
def : Pat<(TF_ConstOp:$res NonOpaqueElementsAttr:$value), (ConstantOp $value),
          [(AnyStaticShapeTensor $res)]>;

// Partially supported in TFLite, treated as passthrough IdentityOp
def : Pat<(TF_CheckNumericsOp $arg, $msg), (TF_IdentityOp $arg)>;
def : Pat<(TF_SnapshotOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_PlaceholderWithDefaultOp $arg), (TF_IdentityOp $arg)>;

//===----------------------------------------------------------------------===//
// Op removal patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_IdentityOp $arg), (replaceWithValue $arg)>;
def : Pat<(TF_IdentityNOp $arg), (replaceWithValue $arg)>;

// Cleanup dead ops manually. LCE ops are not registered to the TF dialect so
// op->hasNoSideEffect() will return false. Therefore
// applyPatternsAndFoldGreedily won't automatically remove the dead nodes. See
// https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/IR/Operation.h#L457-L462
def : Pat<(TF_LceBsignOp:$op $x), (replaceWithValue $x), [(HasNoUseOf:$op)]>;
def : Pat<(TF_LceBconv2dOp:$op
              $input,
              $filter,
              $post_activation_multiplier,
              $post_activation_bias,
              $write_bitpacked_threshold,
              $channels_in,
              $dilation_height_factor,
              $dilation_width_factor,
              $fused_activation_function,
              $pad_values,
              $padding,
              $stride_height,
              $stride_width),
          (replaceWithValue $input), [(HasNoUseOf:$op)]>;
def : Pat<(TF_LceBMaxPool2dOp:$op
              $input,
              $padding,
              $stride_width,
              $stride_height,
              $filter_width,
              $filter_height),
          (replaceWithValue $input), [(HasNoUseOf:$op)]>;
