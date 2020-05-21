// This is the operation definition file for Larq Compute engine ops.

// We extend the TensorFlow dialect in order to allow for easy generation of
// of the TFLite flatbuffer using TensorFlows infrastructure.

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

#ifndef TFL_OPS
def TFL_AF_None  : StrEnumAttrCase<"NONE">;
def TFL_AF_Relu  : StrEnumAttrCase<"RELU">;
def TFL_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TFL_AF_Relu6 : StrEnumAttrCase<"RELU6">;

def TFL_AFAttr : StrEnumAttr<
    "ActivationFunctionType", "fused activation enum", [
      TFL_AF_None,  TFL_AF_Relu, TFL_AF_Relu1, TFL_AF_Relu6
    ]>;
#endif

#ifndef LCE_OPS
#define LCE_OPS

def TF_LceBsignOp : TF_Op<"LceBsign", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Returns an element-wise indication of the binary sign of a number.";

  let description = [{
`y = sign(x) = -1` if `x < 0`; 1 if `x >= 0`.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LceBconv2dOp : TF_Op<"LceBconv2d", [NoSideEffect]> {
  let summary = [{
Computes a 2-D binary convolution by binarizing and bitpacking the input and filter.
  }];

  let description = [{
TODO
  }];

  let arguments = (ins
    TensorOf<[F32]>:$input,
    TensorOf<[F32, I32]>:$filter,
    TensorOf<[F32]>:$post_activation_multiplier,
    TensorOf<[F32]>:$post_activation_bias,

    I32Attr:$channels_in,
    I32Attr:$dilation_height_factor,
    I32Attr:$dilation_width_factor,
    DefaultValuedAttr<TF_AnyStrAttrOf<["OHWI", "OHWI_PACKED"]>, "OHWI">:$filter_format,
    TFL_AFAttr:$fused_activation_function,
    DefaultValuedAttr<I32Attr, "0">:$pad_values,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    I32Attr:$stride_height,
    I32Attr:$stride_width
  );

  let results = (outs
    TensorOf<[F32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_LceBMaxPool2dOp : TF_Op<"LceBMaxPool2d", [NoSideEffect]> {
  let summary = [{
Binary MaxPool2D op.
  }];

  let description = [{
Computes a MaxPool2D operation and outputs bitpacked binary values, for consumption by a binary convolution.
  }];

  let arguments = (ins
    TensorOf<[F32, I32]>:$input,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    I32Attr:$stride_width,
    I32Attr:$stride_height,
    I32Attr:$filter_width,
    I32Attr:$filter_height
  );

  let results = (outs
    TensorOf<[I32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

#endif // LCE_OPS
