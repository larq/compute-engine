#include "larq_compute_engine/mlir/ir/lce_ops.h"

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace TF {

static tflite::Padding ConvertPaddingAttr(llvm::StringRef str) {
  return llvm::StringSwitch<tflite::Padding>(str)
      .Case("SAME", tflite::Padding_SAME)
      .Case("VALID", tflite::Padding_VALID);
}

static tflite::ActivationFunctionType ConvertActivationAttr(
    llvm::StringRef str) {
  return llvm::StringSwitch<tflite::ActivationFunctionType>(str)
      .Case("NONE", tflite::ActivationFunctionType_NONE)
      .Case("RELU", tflite::ActivationFunctionType_RELU)
      .Case("RELU_N1_TO_1", tflite::ActivationFunctionType_RELU_N1_TO_1)
      .Case("RELU6", tflite::ActivationFunctionType_RELU6);
}

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"

std::vector<uint8_t> QuantizeOp::buildCustomOptions() { return {}; }
std::vector<uint8_t> DequantizeOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> Bconv2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("channels_in", channels_in().getSExtValue());
    fbb.Int("dilation_height_factor", dilation_height_factor().getSExtValue());
    fbb.Int("dilation_width_factor", dilation_width_factor().getSExtValue());
    fbb.Int("fused_activation_function",
            (int)ConvertActivationAttr(fused_activation_function()));
    fbb.Int("pad_values", pad_values().getSExtValue());
    fbb.Int("padding", (int)ConvertPaddingAttr(padding()));
    fbb.Int("stride_height", stride_height().getSExtValue());
    fbb.Int("stride_width", stride_width().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> BMaxPool2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("padding", (int)ConvertPaddingAttr(padding()));
    fbb.Int("stride_width", stride_width().getSExtValue());
    fbb.Int("stride_height", stride_height().getSExtValue());
    fbb.Int("filter_width", filter_width().getSExtValue());
    fbb.Int("filter_height", filter_height().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

void QuantizeOp::build(OpBuilder& builder, OperationState& state, Value x) {
  state.addOperands(x);
  const auto existing_shape = x.getType().cast<ShapedType>().getShape();
  const auto channels = existing_shape[existing_shape.size() - 1];
  const auto packed_channels = (channels + 32 - 1) / 32;
  std::vector<int64_t> shape = existing_shape.drop_back();
  shape.push_back(packed_channels);
  state.addTypes(RankedTensorType::get(shape, builder.getIntegerType(32)));
}

LarqDialect::LarqDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"
      >();
}
}  // namespace TF
}  // namespace mlir
