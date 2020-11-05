#ifndef LARQ_COMPUTE_ENGINE_CORE_BCONV2D_PARAMS
#define LARQ_COMPUTE_ENGINE_CORE_BCONV2D_PARAMS

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"

namespace compute_engine {
namespace core {
namespace bconv2d {

struct BConv2DParams {
  // Input and filter shapes
  std::int32_t filter_width;
  std::int32_t filter_height;
  std::int32_t channels_in;
  std::int32_t channels_out;

  // Strides
  std::int32_t stride_height;
  std::int32_t stride_width;

  // Dilations
  std::int32_t dilation_height_factor;
  std::int32_t dilation_width_factor;

  // Padding
  TfLitePadding padding_type;
  TfLitePaddingValues padding_values;
  std::int32_t pad_value;  // Must be 0 or 1
};

}  // namespace bconv2d
}  // namespace core
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_CORE_BCONV2D_PARAMS
