// This is the operation definition file for Larq Compute engine ops.

//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Larq dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"

#ifndef TFL_OP_ENUMS
class TFL_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;
// Allowed activation function cases
// These should match the ActivationFunctionType enum in TFLite schema.
def TFL_AFEnum_None  : I32EnumAttrCase<"NONE", 0>;
def TFL_AFEnum_Relu  : I32EnumAttrCase<"RELU", 1>;
def TFL_AFEnum_Relu1 : I32EnumAttrCase<"RELU_N1_TO_1", 2>;
def TFL_AFEnum_Relu6 : I32EnumAttrCase<"RELU6", 3>;
def TFL_AFEnum_Tanh  : I32EnumAttrCase<"TANH", 4>;
def TFL_AFEnum_Sign  : I32EnumAttrCase<"SIGN_BIT", 5>;

def TFL_AFAttr : TFL_AnyStrAttrOf<[
      TFL_AFEnum_None.symbol,  TFL_AFEnum_Relu.symbol, TFL_AFEnum_Relu1.symbol,
      TFL_AFEnum_Relu6.symbol, TFL_AFEnum_Tanh.symbol, TFL_AFEnum_Sign.symbol
    ]>;

// Allowed padding cases
// These should match the padding enum in TFLite schema.
def TFL_PADEnum_Same  : I32EnumAttrCase<"SAME", 0>;
def TFL_PADEnum_Valid : I32EnumAttrCase<"VALID", 1>;
def TFL_PaddingAttr : TFL_AnyStrAttrOf<[
      TFL_PADEnum_Same.symbol, TFL_PADEnum_Valid.symbol
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

  let hasConstantMaterializer = 1;

  let cppNamespace = "::mlir::lq";
}

//===----------------------------------------------------------------------===//
// Larq op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class LQ_Op<string mnemonic, list<Trait> traits = []> :
    Op<LarqDialect, mnemonic, traits> {

  let extraClassDeclaration = [{
    std::vector<uint8_t> buildCustomOptions();
  }];
}


class TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

def LQ_QuantizeOp : LQ_Op<"Quantize", [Pure]> {
  let summary = "Binary quantize operator";

  let description = [{
Converts floating point, integer, or boolean tensors to binarized bitpacked tensors.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, QI8, QI16, I1]>:$x
  );

  let results = (outs
    TensorOf<[I32]>:$y
  );

  let builders = [OpBuilder<(ins "Value":$x)>];

  let hasFolder = 1;
}

def LQ_DequantizeOp : LQ_Op<"Dequantize", [Pure]> {
  let summary = "Binary dequantize operator";

  let description = [{
Converts binarized bitpacked tensors to floating point, integer, or boolean tensors.
  }];

  let arguments = (ins
    TensorOf<[I32]>:$x
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, QI8, QI16, I1]>:$y
  );

  let hasFolder = 1;
}

def LQ_Bconv2dOp : LQ_Op<"Bconv2d", [Pure]> {
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

def LQ_BMaxPool2dOp : LQ_Op<"BMaxPool2d", [Pure]> {
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
