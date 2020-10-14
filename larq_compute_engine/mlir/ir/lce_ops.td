// This is the operation definition file for Larq Compute engine ops.

//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Larq dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"

#ifndef TFL_OPS
def TFL_AF_None  : StrEnumAttrCase<"NONE">;
def TFL_AF_Relu  : StrEnumAttrCase<"RELU">;
def TFL_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TFL_AF_Relu6 : StrEnumAttrCase<"RELU6">;

def TFL_AFAttr : StrEnumAttr<
    "ActivationFunctionType", "fused activation enum", [
      TFL_AF_None,  TFL_AF_Relu, TFL_AF_Relu1, TFL_AF_Relu6
    ]>;

def TFL_PAD_Same  : StrEnumAttrCase<"SAME">;
def TFL_PAD_Valid : StrEnumAttrCase<"VALID">;

def TFL_PaddingAttr : StrEnumAttr<"Padding", "padding enum", [
      TFL_PAD_Same, TFL_PAD_Valid
    ]>;
#endif

//===----------------------------------------------------------------------===//
// Larq dialect definitions
//===----------------------------------------------------------------------===//

#ifndef LARQ_DIALECT
#define LARQ_DIALECT

def LarqDialect : Dialect {
  let name = "lq";

  let summary = "Types and operations for Larq dialect";
  let description = [{
    This dialect contains operations for Larq. This dialect will be used in
    conjunction with the TensorFlow dialects for converting & optimizing
    TF graphs to be deployed on Larq Compute Engine.
  }];

  let cppNamespace = "::mlir::lq";
}

//===----------------------------------------------------------------------===//
// Larq op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class LQ_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LarqDialect, mnemonic, traits> {

  let extraClassDeclaration = [{
    std::vector<uint8_t> buildCustomOptions();
  }];
}


class TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

def LQ_QuantizeOp : LQ_Op<"Quantize", [NoSideEffect]> {
  let summary = "Binary quantize operator";

  let description = [{
Converts floating point or integer tensors to binarized bitpacked tensors.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, QI8, QI16]>:$x
  );

  let results = (outs
    TensorOf<[I32]>:$y
  );

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &state, Value x"
  >];
}

def LQ_DequantizeOp : LQ_Op<"Dequantize", [NoSideEffect]> {
  let summary = "Binary dequantize operator";

  let description = [{
Converts binarized bitpacked tensors to floating point or integer tensors.
  }];

  let arguments = (ins
    TensorOf<[I32]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, QI8, QI16]>:$y
  );
}

def LQ_Bconv2dOp : LQ_Op<"Bconv2d", [NoSideEffect]> {
  let summary = [{
Computes a 2D binary convolution by binarizing and bitpacking the input and filter.
  }];

  let description = [{
TODO
  }];

  let arguments = (ins
    TensorOf<[I32]>:$input,
    TensorOf<[F32, I32]>:$filter,
    TensorOfOrNone<[F32]>:$post_activation_multiplier,
    TensorOfOrNone<[F32]>:$post_activation_bias,
    TensorOfOrNone<[I32]>:$output_threshold,

    I32Attr:$channels_in,
    I32Attr:$dilation_height_factor,
    I32Attr:$dilation_width_factor,
    TFL_AFAttr:$fused_activation_function,
    DefaultValuedAttr<I32Attr, "0">:$pad_values,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_height,
    I32Attr:$stride_width
  );

  let results = (outs
    TensorOf<[F32, I32, QI8]>:$output
  );
}

def LQ_BMaxPool2dOp : LQ_Op<"BMaxPool2d", [NoSideEffect]> {
  let summary = [{
Binary MaxPool2D op.
  }];

  let description = [{
Computes a MaxPool2D operation and outputs bitpacked binary values, for consumption by a binary convolution.
  }];

  let arguments = (ins
    TensorOf<[I32]>:$input,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_width,
    I32Attr:$stride_height,
    I32Attr:$filter_width,
    I32Attr:$filter_height
  );

  let results = (outs
    TensorOf<[I32]>:$output
  );
}

#endif // LARQ_DIALECT
