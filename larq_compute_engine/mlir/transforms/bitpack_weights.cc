#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace TFL {

namespace {

using compute_engine::core::bitpacking_bitwidth;
using compute_engine::core::TBitpacked;

struct BitpackWeightsLCE : public PassWrapper<BitpackWeightsLCE, FunctionPass> {
  void runOnFunction() override;
};

bool IsConv2DFilter(Attribute filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;
  auto filter_type = filter.getType().cast<ShapedType>();
  return filter_type.getElementType().isF32() &&
         filter_type.getShape().size() == 4;
}

DenseElementsAttr Bitpack(PatternRewriter& builder, Attribute x) {
  const auto& dense_elements_iter =
      x.cast<DenseElementsAttr>().getValues<float>();

  auto shape = x.getType().cast<ShapedType>().getShape();
  int num_rows = shape[0] * shape[1] * shape[2];
  int unpacked_channels = shape[3];
  int packed_channels = compute_engine::core::GetPackedSize(unpacked_channels);

  std::vector<TBitpacked> new_values(num_rows * packed_channels);
  std::vector<float> old_values(num_rows * unpacked_channels);

  int i = 0;
  for (float x : dense_elements_iter) {
    old_values[i++] = x;
  }
  assert(i == num_rows * unpacked_channels);

  using namespace compute_engine::core;
  packbits_matrix(old_values.data(), num_rows, unpacked_channels,
                  new_values.data());

  RankedTensorType out_tensor_type =
      RankedTensorType::get({shape[0], shape[1], shape[2], packed_channels},
                            builder.getIntegerType(bitpacking_bitwidth));

  return DenseElementsAttr::get<TBitpacked>(out_tensor_type, new_values);
}

#include "larq_compute_engine/mlir/transforms/generated_bitpack_weights.inc"

void BitpackWeightsLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect BitpackWeights pass.
std::unique_ptr<OperationPass<FuncOp>> CreateBitpackWeightsLCEPass() {
  return std::make_unique<BitpackWeightsLCE>();
}

static PassRegistration<BitpackWeightsLCE> pass("tfl-lce-bitpack-weights",
                                                "Bitpack binary weights");

}  // namespace TFL
}  // namespace mlir
