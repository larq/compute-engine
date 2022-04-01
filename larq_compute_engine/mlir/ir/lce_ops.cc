#include "larq_compute_engine/mlir/ir/lce_ops.h"

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/mlir/transforms/bitpack.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Generated dialect defs.
#include "larq_compute_engine/mlir/ir/lce_dialect.cc.inc"

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_enum.cc.inc"
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"

namespace mlir {
namespace lq {

std::vector<uint8_t> QuantizeOp::buildCustomOptions() { return {}; }
std::vector<uint8_t> DequantizeOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> Bconv2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("channels_in", channels_in());
    fbb.Int("dilation_height_factor", dilation_height_factor());
    fbb.Int("dilation_width_factor", dilation_width_factor());
    fbb.Int("fused_activation_function",
            (int)symbolizeActivationFunctionType(fused_activation_function())
                .getValue());
    fbb.Int("pad_values", pad_values());
    fbb.Int("padding", (int)symbolizePadding(padding()).getValue());
    fbb.Int("stride_height", stride_height());
    fbb.Int("stride_width", stride_width());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> BMaxPool2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("padding", (int)symbolizePadding(padding()).getValue());
    fbb.Int("stride_width", stride_width());
    fbb.Int("stride_height", stride_height());
    fbb.Int("filter_width", filter_width());
    fbb.Int("filter_height", filter_height());
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

OpFoldResult QuantizeOp::fold(ArrayRef<Attribute> operands) {
  mlir::OpBuilder builder(getOperation());
  if (!operands[0]) return nullptr;
  return mlir::TFL::Bitpack(&builder, operands[0]);
}

OpFoldResult DequantizeOp::fold(ArrayRef<Attribute> operands) {
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
}  // namespace lq
}  // namespace mlir
