include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

class ConstFloatValueIs<string val>
    : Constraint<CPred<
          "$0.isa<DenseElementsAttr>() && "
          "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
          "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == " #val>>;

def NoBatchAndChannelPadding : Constraint<CPred<"NoBatchAndChannelPadding($0)">>;
// TableGen has a maximum of 4 arguments to these constraints, so we have to split the height and the width
def SamePaddingHeight : Constraint<CPred<"IsSamePaddingPartial($0, $1, $2, $3, 1)">>;
def SamePaddingWidth : Constraint<CPred<"IsSamePaddingPartial($0, $1, $2, $3, 2)">>;

// The difference between Pad and PadV2 is the `pad_value` input

// Pad > Conv2D
def : Pat<(TFL_Conv2DOp:$conv_output
                         (TFL_PadOp:$pad_output
                                      $input,
                                      (ConstantOp $paddings)),
                         $filter,
                         $bias,
                         $h_factor,
                         $w_factor,
                         $act_fn,
                         TFL_PAD_Valid,
                         $stride_h,
                         $stride_w),
          (TFL_Conv2DOp $input,
                        $filter,
                        $bias,
                        $h_factor,
                        $w_factor,
                        $act_fn,
                        TFL_PAD_Same,
                        $stride_h,
                        $stride_w),
          [(HasOneUse $pad_output),
           (NoBatchAndChannelPadding $paddings),
           (SamePaddingHeight $paddings, $input, $conv_output, $stride_h),
           (SamePaddingWidth $paddings, $input, $conv_output, $stride_w)],
          (addBenefit 100)>;


// PadV2 > Conv2D
def : Pat<(TFL_Conv2DOp:$conv_output
                         (TFL_PadV2Op:$pad_output
                                      $input,
                                      (ConstantOp $paddings),
                                      (ConstantOp $pad_values)),
                         $filter,
                         $bias,
                         $h_factor,
                         $w_factor,
                         $act_fn,
                         TFL_PAD_Valid,
                         $stride_h,
                         $stride_w),
          (TFL_Conv2DOp $input,
                        $filter,
                        $bias,
                        $h_factor,
                        $w_factor,
                        $act_fn,
                        TFL_PAD_Same,
                        $stride_h,
                        $stride_w),
          [(HasOneUse $pad_output),
           (ConstFloatValueIs<"0.0"> $pad_values),
           (NoBatchAndChannelPadding $paddings),
           (SamePaddingHeight $paddings, $input, $conv_output, $stride_h),
           (SamePaddingWidth $paddings, $input, $conv_output, $stride_w)],
          (addBenefit 100)>;

// Pad > DepthwiseConv2D
def : Pat<(TFL_DepthwiseConv2DOp:$conv_output
                         (TFL_PadOp:$pad_output
                                      $input,
                                      (ConstantOp $paddings)),
                         $filter,
                         $bias,
                         $h_factor,
                         $w_factor,
                         $act_fn,
                         TFL_PAD_Valid,
                         $stride_h,
                         $stride_w,
                         $depth_multiplier),
          (TFL_DepthwiseConv2DOp $input,
                        $filter,
                        $bias,
                        $h_factor,
                        $w_factor,
                        $act_fn,
                        TFL_PAD_Same,
                        $stride_h,
                        $stride_w,
                        $depth_multiplier),
          [(HasOneUse $pad_output),
           (NoBatchAndChannelPadding $paddings),
           (SamePaddingHeight $paddings, $input, $conv_output, $stride_h),
           (SamePaddingWidth $paddings, $input, $conv_output, $stride_w)],
          (addBenefit 100)>;

// PadV2 > DepthwiseConv2D
def : Pat<(TFL_DepthwiseConv2DOp:$conv_output
                         (TFL_PadV2Op:$pad_output
                                      $input,
                                      (ConstantOp $paddings),
                                      (ConstantOp $pad_values)),
                         $filter,
                         $bias,
                         $h_factor,
                         $w_factor,
                         $act_fn,
                         TFL_PAD_Valid,
                         $stride_h,
                         $stride_w,
                         $depth_multiplier),
          (TFL_DepthwiseConv2DOp $input,
                        $filter,
                        $bias,
                        $h_factor,
                        $w_factor,
                        $act_fn,
                        TFL_PAD_Same,
                        $stride_h,
                        $stride_w,
                        $depth_multiplier),
          [(HasOneUse $pad_output),
           (ConstFloatValueIs<"0.0"> $pad_values),
           (NoBatchAndChannelPadding $paddings),
           (SamePaddingHeight $paddings, $input, $conv_output, $stride_h),
           (SamePaddingWidth $paddings, $input, $conv_output, $stride_w)],
          (addBenefit 100)>;
