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

// This file is originally copied from
// "tensorflow/lite/kernels/internal/reference/conv.h".
// However, it's modified to perform a binary convolution instead.

#ifndef COMPUTE_ENGINE_CORE_BCONV2D_REFERENCE_H_
#define COMPUTE_ENGINE_CORE_BCONV2D_REFERENCE_H_

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace bconv2d {

template <typename AccumScalar, typename DstScalar,
          OutputTransformDetails details>
inline void BConv2DReference(
    const BConv2DParams* bconv2d_params, const RuntimeShape& packed_input_shape,
    const TBitpacked* packed_input_data,
    const RuntimeShape& packed_filter_shape,
    const TBitpacked* packed_filter_data,
    const OutputTransform<DstScalar, details>& output_transform,
    const RuntimeShape& output_shape, DstScalar* output_data) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, TBitpacked>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "The reference implementation supports either float "
                "output, bitpacked output or 8-bit quantized output.");

  const int stride_width = bconv2d_params->stride_width;
  const int stride_height = bconv2d_params->stride_height;
  const int dilation_width_factor = bconv2d_params->dilation_width_factor;
  const int dilation_height_factor = bconv2d_params->dilation_height_factor;
  const int pad_width = bconv2d_params->padding_values.width;
  const int pad_height = bconv2d_params->padding_values.height;

  TFLITE_DCHECK_EQ(packed_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(packed_filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(packed_input_shape, 0, output_shape, 0);
  const int input_depth_per_group = packed_filter_shape.Dims(3);
  const int output_depth = packed_filter_shape.Dims(0);
  const int output_depth_per_group = output_depth / bconv2d_params->groups;
  const int input_height = packed_input_shape.Dims(1);
  const int input_width = packed_input_shape.Dims(2);
  const int filter_height = packed_filter_shape.Dims(1);
  const int filter_width = packed_filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const bool zero_padding =
      bconv2d_params->padding_type == kTfLitePaddingSame &&
      bconv2d_params->pad_value == 0;

  // For n channels, a popcount of n/2 of the {0,1} bits would correspond to 0
  // in the {-1,1} representation. So n/2 can be considered the 'zero point'.
  const int binary_zero_point =
      (bconv2d_params->channels_in / bconv2d_params->groups) / 2;

  TFLITE_DCHECK_EQ(input_depth_per_group * bconv2d_params->groups,
                   packed_input_shape.Dims(3));
  TFLITE_DCHECK_EQ(output_depth_per_group * bconv2d_params->groups,
                   output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        // This variable is only used if we are writing bitpacked output.
        TBitpacked bitpacked_column = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          const int group = out_channel / output_depth_per_group;
          AccumScalar accum = AccumScalar(0);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const bool inside = ((in_x >= 0) && (in_x < input_width) &&
                                   (in_y >= 0) && (in_y < input_height));
              if (zero_padding && !inside) {
                accum += binary_zero_point;
                continue;
              }
              for (int in_channel = 0; in_channel < input_depth_per_group;
                   ++in_channel) {
                TBitpacked input_value = 0;  // represents a +1
                if (inside) {
                  input_value = packed_input_data[Offset(
                      packed_input_shape, batch, in_y, in_x,
                      group * input_depth_per_group + in_channel)];
                }
                TBitpacked filter_value =
                    packed_filter_data[Offset(packed_filter_shape, out_channel,
                                              filter_y, filter_x, in_channel)];
                accum += core::xor_popcount(input_value, filter_value);
              }
            }
          }
          // Are we writing bitpacked output?
          if (std::is_same<DstScalar, TBitpacked>::value) {
            bool bit = output_transform.Run(accum, out_channel);
            if (bit) {
              bitpacked_column |= TBitpacked(1)
                                  << (out_channel % bitpacking_bitwidth);
            }

            // After we've 'filled' the `bitpacked_column`, or reached the end
            // of the channels, we write it to memory.
            if ((out_channel + 1) % bitpacking_bitwidth == 0 ||
                (out_channel + 1 == output_depth)) {
              output_data[Offset(output_shape, batch, out_y, out_x,
                                 out_channel / bitpacking_bitwidth)] =
                  bitpacked_column;
              bitpacked_column = 0;
            }
          }
          // Otherwise, we're not writing bitpacked output; it must be int8 or
          // float.
          else {
            DstScalar dst_val = output_transform.Run(accum, out_channel);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               out_channel)] = dst_val;
          }
        }
      }
    }
  }
}

}  // namespace bconv2d
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BCONV2D_REFERENCE_H_
