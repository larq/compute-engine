#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_

#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

struct Kernel {
  const std::int32_t block_size_output_channels;
  const std::int32_t block_size_pixels;
  const std::int32_t block_size_depth;

  const std::int32_t input_depth;
  const std::int32_t output_channels;
  const std::int32_t filter_size;
  const std::int32_t groups;
  const std::int32_t num_output_pixels;

  std::vector<TBitpacked> packed_weights;
  std::vector<const TBitpacked*> indirection_buffer;
  std::vector<TBitpacked> zero_buffer;

  Kernel(const std::int32_t block_size_output_channels,
         const std::int32_t block_size_pixels,
         const std::int32_t block_size_depth,
         const bconv2d::BConv2DParams* bconv2d_params,
         const RuntimeShape& bitpacked_input_shape,
         const RuntimeShape& output_shape)
      : block_size_output_channels(block_size_output_channels),
        block_size_pixels(block_size_pixels),
        block_size_depth(block_size_depth),
        input_depth(bitpacked_input_shape.Dims(3)),
        output_channels(bconv2d_params->channels_out),
        filter_size(bconv2d_params->filter_height *
                    bconv2d_params->filter_width),
        groups(bconv2d_params->groups),
        num_output_pixels(bitpacked_input_shape.Dims(0) * output_shape.Dims(1) *
                          output_shape.Dims(2)) {}

  /**
   * Pack the weights in the correct order. This procedure is (heavily)
   * adapted from the following XNNPack function:
   * https://github.com/google/XNNPACK/blob/80a8ac59849bfdae8d2e1409f5642baa502c0b9e/src/packing.c#L429-L484
   */
  void PackWeights(const TBitpacked* weights_ptr) {
    const std::int32_t input_depth_per_group = input_depth / groups;
    const std::int32_t output_channels_per_group = output_channels / groups;
    const std::int32_t rounded_up_output_channels_per_group =
        block_size_output_channels *
        ((output_channels_per_group + block_size_output_channels - 1) /
         block_size_output_channels);
    packed_weights.resize(groups * rounded_up_output_channels_per_group *
                              filter_size * input_depth_per_group +
                          /* padding */ block_size_output_channels *
                              block_size_depth);
    std::int32_t packed_weights_index = 0;
    for (std::int32_t group_id = 0; group_id < groups; group_id++) {
      for (std::int32_t block_start = 0;
           block_start < output_channels_per_group;
           block_start += block_size_output_channels) {
        const std::int32_t block_size =
            std::min(output_channels_per_group - block_start,
                     block_size_output_channels);
        for (std::int32_t fi = 0; fi < filter_size; fi++) {
          for (std::int32_t ci = 0; ci < input_depth_per_group;
               ci += block_size_depth) {
            for (std::int32_t block_offset = 0; block_offset < block_size;
                 block_offset++) {
              for (std::int32_t ci_offset = 0; ci_offset < block_size_depth;
                   ci_offset++) {
                const std::int32_t weights_index =
                    (group_id * output_channels_per_group * filter_size *
                     input_depth_per_group) +
                    ((block_start + block_offset) * filter_size *
                     input_depth_per_group) +
                    fi * input_depth_per_group + ci + ci_offset;
                packed_weights.at(packed_weights_index++) =
                    weights_ptr[weights_index];
              }
            }
            packed_weights_index +=
                (block_size_output_channels - block_size) * block_size_depth;
          }
        }
      }
    }
  }

  /**
   * Fill the indirection buffer. This procedure is (heavily) adapted from the
   * following XNNPack function:
   * https://github.com/google/XNNPACK/blob/80a8ac59849bfdae8d2e1409f5642baa502c0b9e/src/indirection.c#L18-L76
   */
  void FillIndirectionBuffer(const bconv2d::BConv2DParams* bconv2d_params,
                             const RuntimeShape& bitpacked_input_shape,
                             const RuntimeShape& output_shape,
                             const TBitpacked* input_ptr) {
    const std::int32_t filter_height = bconv2d_params->filter_height;
    const std::int32_t filter_width = bconv2d_params->filter_width;
    const std::int32_t stride_height = bconv2d_params->stride_height;
    const std::int32_t stride_width = bconv2d_params->stride_width;
    const std::int32_t dilation_height = bconv2d_params->dilation_height_factor;
    const std::int32_t dilation_width = bconv2d_params->dilation_width_factor;
    const std::int32_t input_padding_top =
        bconv2d_params->padding_values.height;
    const std::int32_t input_padding_left =
        bconv2d_params->padding_values.width;
    const std::int32_t batch_size = bitpacked_input_shape.Dims(0);
    const std::int32_t input_height = bitpacked_input_shape.Dims(1);
    const std::int32_t input_width = bitpacked_input_shape.Dims(2);
    const std::int32_t output_height = output_shape.Dims(1);
    const std::int32_t output_width = output_shape.Dims(2);
    const std::int32_t output_size = num_output_pixels;
    const std::int32_t tiled_output_size =
        block_size_pixels *
        ((output_size + block_size_pixels - 1) / block_size_pixels);

    // Create the indirection buffer with padding (+ block_size_pixels) and fill
    // it with pointers to the first element of the input, so that the padding
    // at the end of the array contains pointers to valid memory.
    indirection_buffer.assign(
        tiled_output_size * filter_size + block_size_pixels, input_ptr);
    // Assign the zero buffer that will be used for padding.
    zero_buffer.assign(filter_size * input_depth, 0);

    for (std::int32_t output_tile_start = 0;
         output_tile_start < tiled_output_size;
         output_tile_start += block_size_pixels) {
      for (std::int32_t output_tile_offset = 0;
           output_tile_offset < block_size_pixels; output_tile_offset++) {
        const std::int32_t output_index =
            std::min(output_tile_start + output_tile_offset, output_size - 1);
        const std::int32_t batch_index =
            output_index / (output_height * output_width);
        const std::int32_t output_x = output_index % output_width;
        const std::int32_t output_y =
            (output_index % (output_height * output_width)) / output_width;
        for (std::int32_t f_y = 0; f_y < filter_height; f_y++) {
          const std::int32_t input_y = output_y * stride_height +
                                       f_y * dilation_height -
                                       input_padding_top;
          if (0 <= input_y && input_y < input_height) {
            for (std::int32_t f_x = 0; f_x < filter_width; f_x++) {
              const std::int32_t input_x = output_x * stride_width +
                                           f_x * dilation_width -
                                           input_padding_left;
              const std::int32_t kernel_index = f_y * filter_width + f_x;
              const std::int32_t index = output_tile_start * filter_size +
                                         kernel_index * block_size_pixels +
                                         output_tile_offset;
              if (0 <= input_x && input_x < input_width) {
                indirection_buffer.at(index) =
                    (input_ptr + (batch_index * input_height * input_width +
                                  input_y * input_width + input_x) *
                                     input_depth);
              } else {
                indirection_buffer.at(index) = zero_buffer.data();
              }
            }
          } else {
            for (std::int32_t f_x = 0; f_x < filter_width; f_x++) {
              const std::int32_t kernel_index = f_y * filter_width + f_x;
              const std::int32_t index = output_tile_start * filter_size +
                                         kernel_index * block_size_pixels +
                                         output_tile_offset;
              indirection_buffer.at(index) = zero_buffer.data();
            }
          }
        }
      }
    }
  }

  // To be implemented by concrete subclasses.
  virtual void Run(const std::int32_t pixel_start, const std::int32_t pixel_end,
                   void* output_ptr) const = 0;

  void Dispatch(void* output_ptr) const {
    // TODO: implement multithreading here.
    Run(0, num_output_pixels, output_ptr);
  };

  virtual ~Kernel() {}
};

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_
