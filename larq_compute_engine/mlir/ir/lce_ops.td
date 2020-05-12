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

    DefaultValuedAttr<I32Attr, "0">:$channels_in,
    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<I32Attr, "0">:$pad_values,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<TF_AnyStrAttrOf<["OHWI", "OHWI_PACKED"]>, "OHWI">:$filter_format,
    TFL_AFAttr:$activation
  );

  let results = (outs
    TensorOf<[F32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
