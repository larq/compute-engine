#include "larq_compute_engine/mlir/ir/lce_ops.h"

#include "flatbuffers/flexbuffers.h"

namespace mlir {
namespace TF {

#define GET_OP_CLASSES
#include "larq_compute_engine/mlir/ir/lce_ops.cc.inc"

std::vector<uint8_t> BsignOp::buildCustomOptions() { return {}; }

std::vector<uint8_t> Bconv2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("channels_in", channels_in().getSExtValue());
    fbb.Int("dilation_height_factor", dilation_height_factor().getSExtValue());
    fbb.Int("dilation_width_factor", dilation_width_factor().getSExtValue());
    fbb.String("fused_activation_function",
               std::string(fused_activation_function()));
    fbb.Int("pad_values", pad_values().getSExtValue());
    fbb.String("padding", std::string(padding()));
    fbb.Int("stride_height", stride_height().getSExtValue());
    fbb.Int("stride_width", stride_width().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> BMaxPool2dOp::buildCustomOptions() {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.String("padding", std::string(padding()));
    fbb.Int("stride_width", stride_width().getSExtValue());
    fbb.Int("stride_height", stride_height().getSExtValue());
    fbb.Int("filter_width", filter_width().getSExtValue());
    fbb.Int("filter_height", filter_height().getSExtValue());
  });
  fbb.Finish();
  return fbb.GetBuffer();
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
