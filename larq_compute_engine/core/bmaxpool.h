#ifndef COMPUTE_ENGINE_CORE_BMAXPOOL_H_
#define COMPUTE_ENGINE_CORE_BMAXPOOL_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;
namespace ref {

struct BMaxPoolParams {
  std::int32_t filter_height{0};
  std::int32_t filter_width{0};
  std::int32_t stride_height{0};
  std::int32_t stride_width{0};
  TfLitePaddingValues padding{};
  TfLitePadding padding_type{};
};

// Effectively takes the AND of everything in the filter region
template <typename TBitpacked>
void BMaxPool(const BMaxPoolParams& params, const RuntimeShape& input_shape,
              const TBitpacked* input_data, const RuntimeShape& output_shape,
              TBitpacked* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int filter_height = params.filter_height;
  const int filter_width = params.filter_width;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int channels = MatchingDim(input_shape, 3, output_shape, 3);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - params.padding.width;
        const int in_y_origin = (out_y * stride_height) - params.padding.height;
        // Compute the boundaries of the filter region clamped so as to
        // ensure that the filter window fits in the input array.
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);

        const int in_x = in_x_origin + filter_x_start;
        const int in_y = in_y_origin + filter_y_start;

        const int filter_x_count =
            std::min(filter_width - filter_x_start, input_width - in_x);
        const int filter_y_count =
            std::min(filter_height - filter_y_start, input_height - in_y);

        // How far to jump to the next input pixel in the x direction
        const int x_stride = channels;
        // How far to jump to the next input pixel in the y direction, and
        // 'back' to the first pixel in the x direction.
        const int y_stride = channels * (input_width - filter_x_count);

        // Get the pointer to the input pixel corresponding top-left filter
        // corner, channel 0
        const TBitpacked* in_base =
            &input_data[Offset(input_shape, batch, in_y, in_x, 0)];
        TBitpacked* out_ptr =
            &output_data[Offset(output_shape, batch, out_y, out_x, 0)];

        for (int channel = 0; channel < channels; ++channel) {
          const TBitpacked* in_ptr = in_base + channel;

          // Start with all ones
          TBitpacked max = ~TBitpacked(0);
          for (int y = 0; y < filter_y_count; ++y) {
            for (int x = 0; x < filter_x_count; ++x) {
              max &= *in_ptr;
              in_ptr += x_stride;
            }
            in_ptr += y_stride;
          }
          *out_ptr++ = max;
        }
      }
    }
  }
}

}  // namespace ref
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_MAXPOOL_H_
