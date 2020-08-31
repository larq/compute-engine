#ifndef LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_PARAMS
#define LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_PARAMS

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"

namespace compute_engine {
namespace tflite {
namespace bconv2d {

const int kTensorNotAllocated = -1;

struct TfLiteBConv2DParams {
  // Filters tensor dimensions
  std::int32_t filter_width{0};
  std::int32_t filter_height{0};
  std::int32_t channels_in{0};
  std::int32_t channels_out{0};

  // Strides
  std::int32_t stride_height{0};
  std::int32_t stride_width{0};

  // Dilations
  std::int32_t dilation_height_factor{0};
  std::int32_t dilation_width_factor{0};

  // _adding
  TfLitePadding padding_type{};
  TfLitePaddingValues padding_values{};
  int pad_value = 0;  // Must be 0 or 1

  // These min/max take care of a Relu.
  // Later they will *also* do the clamping in order to go from int32 to int8
  std::int32_t output_activation_min;
  std::int32_t output_activation_max;

  // This is only for int8 mode; it's the post_activation_ values scaled by the
  // output tensor scale, and the bias includes the output zero-point.
  std::vector<float> scaled_post_activation_multiplier;
  std::vector<float> scaled_post_activation_bias;

  // This is used when we have 'same-zero' padding.
  std::vector<float> padding_buffer;

  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers. They are unique in the entire TF Lite context.
  int im2col_id = kTensorNotAllocated;
  // In node->temporaries there is a list of tensor id's that are part
  // of this node in particular. The indices below are offsets into this array.
  // So in pseudo-code: `node->temporaries[index] = id;`
  std::int32_t im2col_index = -1;

  bool conv_params_successfully_initialized = false;

  bool one_time_setup_complete = false;
};

}  // namespace bconv2d
}  // namespace tflite
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_PARAMS
