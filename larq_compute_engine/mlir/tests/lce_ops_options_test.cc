#include <gtest/gtest.h>

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace mlir;
using namespace tflite;

IntegerAttr getIntegerAttr(Builder builder, int value) {
  return builder.getIntegerAttr(builder.getIntegerType(32), value);
}

TEST(LCEOpsSerializationTest, QuantizeTest) {
  MLIRContext context;
  context.getOrLoadDialect<lq::LarqDialect>();
  OperationState state(UnknownLoc::get(&context),
                       OperationName("lq.Quantize", &context));
  mlir::Operation* op = Operation::create(state);

  ASSERT_EQ(cast<lq::QuantizeOp>(op).buildCustomOptions().size(), 0);
}

TEST(LCEOpsSerializationTest, DequantizeTest) {
  MLIRContext context;
  context.getOrLoadDialect<lq::LarqDialect>();
  OperationState state(UnknownLoc::get(&context),
                       OperationName("lq.Dequantize", &context));
  mlir::Operation* op = Operation::create(state);

  ASSERT_EQ(cast<lq::DequantizeOp>(op).buildCustomOptions().size(), 0);
}

TEST(LCEOpsSerializationTest, BConv2dTest) {
  MLIRContext context;
  context.getOrLoadDialect<lq::LarqDialect>();
  Builder builder(&context);
  OperationState state(UnknownLoc::get(&context),
                       OperationName("lq.Bconv2d", &context));
  mlir::Operation* op = Operation::create(state);

  op->setAttr("channels_in", getIntegerAttr(builder, 64));
  op->setAttr("dilation_height_factor", getIntegerAttr(builder, 3));
  op->setAttr("dilation_width_factor", getIntegerAttr(builder, 4));
  op->setAttr("stride_height", getIntegerAttr(builder, 1));
  op->setAttr("stride_width", getIntegerAttr(builder, 2));
  op->setAttr("pad_values", getIntegerAttr(builder, 1));

  op->setAttr("fused_activation_function", builder.getStringAttr("RELU"));
  op->setAttr("padding", builder.getStringAttr("SAME"));

  std::vector<uint8_t> v = cast<lq::Bconv2dOp>(op).buildCustomOptions();
  const flexbuffers::Map& m = flexbuffers::GetRoot(v).AsMap();

  ASSERT_EQ(m["channels_in"].AsInt32(), 64);
  ASSERT_EQ(m["dilation_height_factor"].AsInt32(), 3);
  ASSERT_EQ(m["dilation_width_factor"].AsInt32(), 4);
  ASSERT_EQ(m["stride_height"].AsInt32(), 1);
  ASSERT_EQ(m["stride_width"].AsInt32(), 2);
  ASSERT_EQ(m["pad_values"].AsInt32(), 1);
  ASSERT_EQ((ActivationFunctionType)m["fused_activation_function"].AsInt32(),
            ActivationFunctionType_RELU);
  ASSERT_EQ((Padding)m["padding"].AsInt32(), Padding_SAME);
}

TEST(LCEOpsSerializationTest, BMaxPool2dTest) {
  MLIRContext context;
  context.getOrLoadDialect<lq::LarqDialect>();
  Builder builder(&context);
  OperationState state(UnknownLoc::get(&context),
                       OperationName("lq.BMaxPool2d", &context));
  mlir::Operation* op = Operation::create(state);

  op->setAttr("padding", builder.getStringAttr("SAME"));
  op->setAttr("stride_width", getIntegerAttr(builder, 2));
  op->setAttr("stride_height", getIntegerAttr(builder, 1));
  op->setAttr("filter_width", getIntegerAttr(builder, 3));
  op->setAttr("filter_height", getIntegerAttr(builder, 4));

  std::vector<uint8_t> v = cast<lq::BMaxPool2dOp>(op).buildCustomOptions();
  const flexbuffers::Map& m = flexbuffers::GetRoot(v).AsMap();

  ASSERT_EQ((Padding)m["padding"].AsInt32(), Padding_SAME);
  ASSERT_EQ(m["stride_width"].AsInt32(), 2);
  ASSERT_EQ(m["stride_height"].AsInt32(), 1);
  ASSERT_EQ(m["filter_width"].AsInt32(), 3);
  ASSERT_EQ(m["filter_height"].AsInt32(), 4);
}
