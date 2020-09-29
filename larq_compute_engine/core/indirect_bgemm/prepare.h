#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_PREPARE_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_PREPARE_H_

#include <cstdint>

#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_params.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

// This function is (heavily) adapted from this XNNPack function:
// https://github.com/google/XNNPACK/blob/80a8ac59849bfdae8d2e1409f5642baa502c0b9e/src/indirection.c#L18-L76
void FillIndirectionBuffer(const int block_size_pixels,
                           const TfLiteBConv2DParams* conv_params,
                           const RuntimeShape& bitpacked_input_shape,
                           const RuntimeShape& output_shape,
                           const TBitpacked* input_ptr,
                           std::vector<const TBitpacked*>& indirection_buffer,
                           std::vector<TBitpacked>& zero_buffer) {
  using std::int32_t;

  const int32_t kernel_height = conv_params->filter_height;
  const int32_t kernel_width = conv_params->filter_width;
  const int32_t stride_height = conv_params->stride_height;
  const int32_t stride_width = conv_params->stride_width;
  const int32_t dilation_height = conv_params->dilation_height_factor;
  const int32_t dilation_width = conv_params->dilation_width_factor;
  const int32_t input_padding_top = conv_params->padding_values.height;
  const int32_t input_padding_left = conv_params->padding_values.width;

  const int32_t input_height = bitpacked_input_shape.Dims(1);
  const int32_t input_width = bitpacked_input_shape.Dims(2);
  const int32_t bitpacked_input_channels = bitpacked_input_shape.Dims(3);

  const int32_t output_height = output_shape.Dims(1);
  const int32_t output_width = output_shape.Dims(2);

  const int32_t output_size = output_height * output_width;
  const int32_t kernel_size = kernel_height * kernel_width;
  const int32_t tiled_output_size =
      block_size_pixels *
      ((output_size + block_size_pixels - 1) / block_size_pixels);

  indirection_buffer.resize(tiled_output_size * kernel_size);
  zero_buffer.assign(kernel_size * bitpacked_input_channels, 0);

  for (int32_t output_tile_start = 0; output_tile_start < tiled_output_size;
       output_tile_start += block_size_pixels) {
    for (int32_t output_tile_offset = 0; output_tile_offset < block_size_pixels;
         output_tile_offset++) {
      const int32_t output_index =
          std::min(output_tile_start + output_tile_offset, output_size - 1);
      const int32_t output_x = output_index % output_width;
      const int32_t output_y = output_index / output_width;
      for (int32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const int32_t input_y = output_y * stride_height +
                                kernel_y * dilation_height - input_padding_top;
        if (0 <= input_y && input_y < input_height) {
          for (int32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const int32_t input_x = output_x * stride_width +
                                    kernel_x * dilation_width -
                                    input_padding_left;
            const int32_t kernel_index = kernel_y * kernel_width + kernel_x;
            const int32_t index = output_tile_start * kernel_size +
                                  kernel_index * block_size_pixels +
                                  output_tile_offset;
            if (0 <= input_x && input_x < input_width) {
              indirection_buffer.at(index) =
                  (input_ptr + (input_y * input_width + input_x) *
                                   bitpacked_input_channels);
            } else {
              indirection_buffer.at(index) = zero_buffer.data();
            }
          }
        } else {
          for (int32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const int32_t kernel_index = kernel_y * kernel_width + kernel_x;
            const int32_t index = output_tile_start * kernel_size +
                                  kernel_index * block_size_pixels +
                                  output_tile_offset;
            indirection_buffer.at(index) = zero_buffer.data();
          }
        }
      }
    }
  }
}

// This function is (heavily) adapted from this XNNPack function:
// https://github.com/google/XNNPACK/blob/80a8ac59849bfdae8d2e1409f5642baa502c0b9e/src/packing.c#L429-L484
void PackWeights(const int block_size_output_channels,
                 const TfLiteBConv2DParams* conv_params,
                 const RuntimeShape& bitpacked_input_shape,
                 const RuntimeShape& output_shape,
                 const TBitpacked* weights_ptr,
                 std::vector<TBitpacked>& packed_weights) {
  using std::int32_t;

  const int32_t bitpacked_input_channels = bitpacked_input_shape.Dims(3);
  const int32_t output_channels = conv_params->channels_out;
  const int32_t kernel_size =
      conv_params->filter_height * conv_params->filter_width;

  const int32_t rounded_up_output_channels =
      block_size_output_channels *
      ((output_channels + block_size_output_channels - 1) /
       block_size_output_channels);

  packed_weights.resize(rounded_up_output_channels * kernel_size *
                        bitpacked_input_channels);

  int32_t packed_weights_index = 0;

  for (int32_t block_start = 0; block_start < output_channels;
       block_start += block_size_output_channels) {
    const int32_t block_size =
        std::min(output_channels - block_start, block_size_output_channels);
    for (int32_t ki = 0; ki < kernel_size; ki++) {
      for (int32_t ci = 0; ci < bitpacked_input_channels; ci++) {
        for (int32_t block_offset = 0; block_offset < block_size;
             block_offset++) {
          const int32_t weights_index = (block_start + block_offset) *
                                            kernel_size *
                                            bitpacked_input_channels +
                                        ki * bitpacked_input_channels + ci;
          packed_weights.at(packed_weights_index++) =
              weights_ptr[weights_index];
        }
        packed_weights_index += block_size_output_channels - block_size;
      }
    }
  }
}

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_PREPARE_H_
