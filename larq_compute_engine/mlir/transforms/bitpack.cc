#include "larq_compute_engine/mlir/transforms/bitpack.h"

#include <cmath>
#include <vector>

#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/types.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

namespace mlir {
namespace TFL {

using compute_engine::core::bitpacking_bitwidth;
using compute_engine::core::round;
using compute_engine::core::saturate;
using compute_engine::core::TBitpacked;
using namespace compute_engine::core::bitpacking;

DenseElementsAttr Bitpack(mlir::Builder* builder, Attribute x) {
  if (!x) return nullptr;

  // ShapedType is something like tensor<1x2x3xf32> and element_type is f32
  auto shaped_type = x.cast<TypedAttr>().getType().cast<ShapedType>();
  auto shape = shaped_type.getShape();
  auto element_type = shaped_type.getElementType();

  int num_rows = shape[0] * shape[1] * shape[2];
  int unpacked_channels = shape[3];
  int packed_channels = GetBitpackedSize(unpacked_channels);

  std::vector<TBitpacked> new_values(num_rows * packed_channels);

  if (element_type.isF32()) {
    const auto& dense_elements_iter =
        x.cast<DenseElementsAttr>().getValues<float>();

    std::vector<float> old_values(num_rows * unpacked_channels);

    int i = 0;
    for (float x : dense_elements_iter) {
      old_values[i++] = x;
    }
    assert(i == num_rows * unpacked_channels);

    bitpack_matrix(old_values.data(), num_rows, unpacked_channels,
                   new_values.data());
  } else {
    // constant-fold bitpacking int8 tensors is currently not supported
    return nullptr;
  }

  RankedTensorType out_tensor_type =
      RankedTensorType::get({shape[0], shape[1], shape[2], packed_channels},
                            builder->getIntegerType(bitpacking_bitwidth));

  return DenseElementsAttr::get<TBitpacked>(out_tensor_type, new_values);
}

DenseElementsAttr Unpack(Attribute x, ShapedType result_type) {
  if (!x) return nullptr;
  if (!result_type.hasStaticShape()) return nullptr;

  auto input_shape =
      x.cast<TypedAttr>().getType().cast<ShapedType>().getShape();
  auto output_shape = result_type.getShape();
  auto output_type = result_type.getElementType();

  int num_rows = output_shape[0] * output_shape[1] * output_shape[2];
  int unpacked_channels = output_shape[3];
  int packed_channels = GetBitpackedSize(unpacked_channels);
  if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1] ||
      input_shape[2] != output_shape[2] || input_shape[3] != packed_channels) {
    return nullptr;
  }

  std::vector<TBitpacked> old_values(num_rows * packed_channels);

  const auto& dense_elements_iter =
      x.cast<DenseElementsAttr>().getValues<TBitpacked>();

  int i = 0;
  for (TBitpacked x : dense_elements_iter) {
    old_values[i++] = x;
  }
  assert(i == num_rows * packed_channels);

  if (output_type.isF32()) {
    std::vector<float> new_values(num_rows * unpacked_channels);

    unpack_matrix(old_values.data(), num_rows, unpacked_channels,
                  new_values.data());

    return DenseElementsAttr::get<float>(result_type, new_values);
  } else {
    auto quant_type = output_type.cast<mlir::quant::UniformQuantizedType>();
    const double scale = quant_type.getScale();
    const int zero_point = quant_type.getZeroPoint();

    std::int8_t zero_bit_result = saturate(zero_point + round(+1.0 / scale));
    std::int8_t one_bit_result = saturate(zero_point + round(-1.0 / scale));

    std::vector<std::int8_t> new_values(num_rows * unpacked_channels);

    unpack_matrix(old_values.data(), num_rows, unpacked_channels,
                  new_values.data(), zero_bit_result, one_bit_result);

    return DenseElementsAttr::get<std::int8_t>(result_type, new_values);
  }
}

}  // namespace TFL
}  // namespace mlir
