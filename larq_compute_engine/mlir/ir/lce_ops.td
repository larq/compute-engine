// This is the operation definition file for Larq Compute engine ops.

// We extend the TensorFlow dialect in order to allow for easy generation of
// of the TFLite flatbuffer using TensorFlows infrastructure.

#ifndef TF_OP_BASE
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
#endif

def TF_LqceBsignOp : TF_Op<"LqceBsign", [NoSideEffect, SameOperandsAndResultType]> {
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

def TF_LqceBconv2d64Op : TF_Op<"LqceBconv2d64", [NoSideEffect]> {
  let summary = [{
Computes a 2-D binary convolution by binarizing and bitpacking the input and filter.
  }];

  let description = [{
TODO
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32]>:$input,
    TensorOf<[BF16, F16, F32, F64, I32]>:$filter,
    TensorOf<[F32]>:$fused_multiply,
    TensorOf<[F32]>:$fused_add,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations,
    DefaultValuedAttr<TF_AnyStrAttrOf<["HWIO", "OHWI", "OHWI_PACKED"]>, "HWIO">:$filter_format
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  /* let verifier = [{
    return Verify(*this);
  }]; */
}
