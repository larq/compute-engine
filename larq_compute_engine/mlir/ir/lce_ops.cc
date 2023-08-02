#include "larq_compute_engine/mlir/ir/lce_ops.h"

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/mlir/transforms/bitpack.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Generated dialect defs.
#include "larq_compute_engine/mlir/ir/lce_dialect.cc.inc"

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

namespace mlir {
namespace lq {

std::vector<uint8_t> QuantizeOp::buildCustomOptions() { return {}; }
std::vector<uint8_t> DequantizeOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> Bconv2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("channels_in", getChannelsIn());
    fbb.Int("dilation_height_factor", getDilationHeightFactor());
    fbb.Int("dilation_width_factor", getDilationWidthFactor());
    fbb.Int("fused_activation_function",
            (int)ConvertActivationAttr(getFusedActivationFunction()));
    fbb.Int("pad_values", getPadValues());
    fbb.Int("padding", (int)ConvertPaddingAttr(getPadding()));
    fbb.Int("stride_height", getStrideHeight());
    fbb.Int("stride_width", getStrideWidth());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> BMaxPool2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("padding", (int)ConvertPaddingAttr(getPadding()));
    fbb.Int("stride_width", getStrideWidth());
    fbb.Int("stride_height", getStrideHeight());
    fbb.Int("filter_width", getFilterWidth());
    fbb.Int("filter_height", getFilterHeight());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

void QuantizeOp::build(OpBuilder& builder, OperationState& state, Value x) {
  state.addOperands(x);
  const auto existing_shape = x.getType().cast<ShapedType>().getShape();
  const auto channels = existing_shape[existing_shape.size() - 1];
  std::vector<int64_t> shape = existing_shape.drop_back();
  shape.push_back(compute_engine::core::bitpacking::GetBitpackedSize(channels));
  state.addTypes(RankedTensorType::get(shape, builder.getIntegerType(32)));
}

OpFoldResult QuantizeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  mlir::OpBuilder builder(getOperation());
  if (!operands[0]) return nullptr;
  return mlir::TFL::Bitpack(&builder, operands[0]);
}

OpFoldResult DequantizeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto result_type = getType().cast<ShapedType>();
  if (!operands[0]) return nullptr;
  return mlir::TFL::Unpack(operands[0], result_type);
}

void LarqDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"
      >();
}

Operation* LarqDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type,
                                                   cast<TypedAttr>(value));
  return nullptr;
}
}  // namespace lq
}  // namespace mlir
