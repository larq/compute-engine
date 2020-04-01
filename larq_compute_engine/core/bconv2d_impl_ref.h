/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Modifications copyright (C) 2020 Larq Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef COMPUTE_EGNINE_CORE_BCONV2d_IMPL_H_
#define COMPUTE_EGNINE_CORE_BCONV2d_IMPL_H_

#include "larq_compute_engine/core/bgemm_functor.h"
#include "larq_compute_engine/core/packbits_utils.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

// This file is originally copied from
// "tensorflow/lite/kernels/internal/reference/conv.h".
// However, it's modified to perform binary convolution instead
using namespace tflite;

namespace compute_engine {
namespace ce = compute_engine;
namespace ref {

template <typename T, typename TBitpacked, typename AccumScalar,
          typename DstScalar>
inline void BConv2D(const ConvParams& params,
                    const RuntimeShape& packed_input_shape,
                    const TBitpacked* packed_input_data,
                    const RuntimeShape& packed_filter_shape,
                    const TBitpacked* packed_filter_data,
                    const float* post_activation_multiplier_data,
                    const float* post_activation_bias_data,
                    const RuntimeShape& output_shape, DstScalar* output_data,
                    const RuntimeShape& im2col_shape, T* im2col_data,
                    bool bitpack_before_im2col, T* padding_buffer,
                    const int pad_value, void* cpu_backend_context,
                    const std::int32_t backtransform_add) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int32_t>::value,
                "The reference implementation supports either float "
                "output or 32-bit bitpacked output.");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  const auto* post_activation_multiplier = post_activation_multiplier_data;
  const auto* post_activation_bias = post_activation_bias_data;
  AccumScalar clamp_min = params.quantized_activation_min;
  AccumScalar clamp_max = params.quantized_activation_max;

  TFLITE_DCHECK_EQ(packed_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(packed_filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(packed_input_shape, 0, output_shape, 0);
  const int input_depth =
      MatchingDim(packed_input_shape, 3, packed_filter_shape, 3);
  const int output_depth = packed_filter_shape.Dims(0);
  const int input_height = packed_input_shape.Dims(1);
  const int input_width = packed_input_shape.Dims(2);
  const int filter_height = packed_filter_shape.Dims(1);
  const int filter_width = packed_filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        // This variable is only used if we are writing bitpacked output.
        std::uint32_t bitpacked_column = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          AccumScalar accum = AccumScalar(0);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // `pad_value=1`, which means the bitpacked value is 0, so we
                // set `input_value=0`
                TBitpacked input_value = 0;
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  input_value = packed_input_data[Offset(
                      packed_input_shape, batch, in_y, in_x, in_channel)];
                }
                TBitpacked filter_value =
                    packed_filter_data[Offset(packed_filter_shape, out_channel,
                                              filter_y, filter_x, in_channel)];
                accum += ce::core::xor_popcount<TBitpacked, AccumScalar>(
                    input_value, filter_value);
              }
            }
          }
          // Backtransformation to {-1,1} space
          accum = backtransform_add - 2 * accum;
          // Activation
          accum = std::min<AccumScalar>(accum, clamp_max);
          accum = std::max<AccumScalar>(accum, clamp_min);
          // Post multiply and add are done in float
          float dst_val = static_cast<float>(accum);
          if (post_activation_multiplier) {
            dst_val *= post_activation_multiplier[out_channel];
          }
          if (post_activation_bias) {
            dst_val += post_activation_bias[out_channel];
          }

          // If the destination scalar is int32, we're writing bitpacked output.
          if (std::is_same<DstScalar, std::int32_t>::value) {
            // In our bitpacking we map strictly negative values to 1, and
            // non-negative values to 0.
            if (dst_val < 0) bitpacked_column += 1 << (out_channel % 32);

            // After we've 'filled' the `bitpacked_column` with 32 values, or
            // reached the end of the channels, we write it to memory.
            if ((out_channel + 1) % 32 == 0 ||
                (out_channel + 1 == output_depth)) {
              output_data[Offset(output_shape, batch, out_y, out_x,
                                 out_channel / 32)] = bitpacked_column;
              bitpacked_column = 0;
            }
          }

          // Otherwise, we're not writing bitpacked output; it must be float.
          else {
            output_data[Offset(output_shape, batch, out_y, out_x,
                               out_channel)] = dst_val;
          }
        }
      }
    }
  }
}

}  // namespace ref
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_CORE_BCONV2d_IMPL_H_
