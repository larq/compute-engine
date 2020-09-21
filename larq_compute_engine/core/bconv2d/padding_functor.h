#ifndef COMPUTE_ENGINE_CORE_BCONV2D_PADDING_FUNCTOR_H_
#define COMPUTE_ENGINE_CORE_BCONV2D_PADDING_FUNCTOR_H_

#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace compute_engine {
namespace core {
namespace bconv2d {

// Applies (in-place) corrections for zero-padding
// Assumes that padding type is 'SAME'.
class PaddingFunctor {
 public:
  static std::size_t get_cache_size(const int filter_height,
                                    const int filter_width,
                                    const int filter_count,
                                    const int dilation_rows,
                                    const int dilation_cols) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    return 4 * effective_filter_height * effective_filter_width * filter_count;
  }

  static void cache_correction_values(
      const TBitpacked* filter_data, const int filter_height,
      const int filter_width, const int filter_count, const int input_channels,
      const int dilation_rows, const int dilation_cols,
      const float* post_activation_multiplier_data, float* output_cache) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    // Given an (out_x, out_y) for which you want to compute the correction:
    //
    // Define:
    // overflow_left  = filter_left_offset - out_x * stride_cols
    // overflow_top   = filter_top_offset - out_y * stride_rows
    // overflow_right = -overflow_left + effective_filter_width - input_width
    // overflow_bot   = -overflow_top + effective_filter_height - input_height
    //
    // These numbers say (for a particular out_x,out_y) how many pixels the
    // kernel sticks out at each side.
    //
    // The correction that we need to apply depends on the overflow_ values.
    //
    // The correction is a sum over values in the kernel.
    // Now we describe which kernel pixels contribute to this correction.
    //
    // Define:
    // effective_filter_x = dilation_rows * filter_x
    // effective_filter_y = dilation_cols * filter_y
    //
    // So fix a (overflow_left,overflow_top,overflow_right,overflow_bot) tuple
    // and loop over filter_x, filter_y.
    // If one of the 4 following are satisfied then this filter value is counted
    // for this tuple as correction.
    //
    // (T)                           effective_filter_y <  overflow_top
    // (B) effective_filter_height - effective_filter_y <= overflow_bot
    // (L)                           effective_filter_x <  overflow_left
    // (R)  effective_filter_width - effective_filter_x <= overflow_right
    //
    // For example when (L) is satisfied then this particular pixel in the
    // filter sticks out at the left.
    //
    // So we sum this over all kernel pixels that stick out somewhere
    // to obtain our correction value.
    //
    // So we could save this value for every tuple.
    // However (T) and (B) cannot be both true at the same time,
    // and neither can (L) and (R) because we will assume that the (effective)
    // filter size is always smaller than the image.
    //
    // We have the following cases:
    //
    // 000011
    // 000011
    // 22..11
    // 22..11
    // 223333
    // 223333
    //                       overflow_
    //           | left    right   top     bot
    // ----------|-----------------------------
    // Case 0    | *       <= 0    >  0    <  0
    // Case 1    | <  0    >  0    *       <= 0
    // Case 2    | >  0    <  0    <= 0    *
    // Case 3    | <= 0    *       <  0    >  0
    //
    // For pixels in the top middle, we have overflow_left < 0,
    // but they all share the same correction, namely the
    // correction for overflow_left = 0.
    // The same happens for every * in the table.
    //
    // Therefore, if there is a * then we should clamp it to 0.
    //

    for (int y = 0; y < effective_filter_height; ++y) {
      for (int x = 0; x < effective_filter_width; ++x) {
        for (int out_c = 0; out_c < filter_count; ++out_c) {
          float corrections[4] = {0, 0, 0, 0};

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              // Sum over input channels
              int popcount = 0;
              int packed_channels =
                  bitpacking::GetBitpackedSize(input_channels);
              for (int in_c = 0; in_c < packed_channels; ++in_c) {
                int filter_idx;
                // filter_data has shape
                // [out_channels, height, width, packed_in_channels]
                filter_idx =
                    out_c * (filter_height * filter_width * packed_channels) +
                    filter_y * (filter_width * packed_channels) +
                    filter_x * packed_channels + in_c;

                // filter = +1.0 --> bit = 0 ; correction += +1.0
                // filter = -1.0 --> bit = 1 ; correction += -1.0
                popcount += xor_popcount(filter_data[filter_idx], 0);
              }
              float cur_correction = input_channels - 2 * popcount;

              const int effective_filter_x = dilation_cols * filter_x;
              const int effective_filter_y = dilation_rows * filter_y;
              // Case 0: overflow_left = x; overflow_top = y
              if (effective_filter_y < y || effective_filter_x < x) {
                corrections[0] += cur_correction;
              }
              // Case 1: overflow_right = x; overflow_top = y
              if (effective_filter_y < y ||
                  (effective_filter_width - effective_filter_x) <= x) {
                corrections[1] += cur_correction;
              }
              // Case 2: overflow_left = x; overflow_bot = y
              if ((effective_filter_height - effective_filter_y) <= y ||
                  effective_filter_x < x) {
                corrections[2] += cur_correction;
              }
              // Case 3: overflow_right = x; overflow_bot = y
              if ((effective_filter_height - effective_filter_y) <= y ||
                  (effective_filter_width - effective_filter_x) <= x) {
                corrections[3] += cur_correction;
              }
            }
          }

          // Output cache size:
          // 4 * effective_height * effective_width * filter_count
          for (int direction = 0; direction < 4; ++direction) {
            int output_idx =
                direction * (effective_filter_height * effective_filter_width *
                             filter_count) +
                y * (effective_filter_width * filter_count) + x * filter_count +
                out_c;
            const float mul = -1.0f * post_activation_multiplier_data[out_c];
            output_cache[output_idx] = mul * corrections[direction];
          }
        }
      }
    }
    return;
  }

  void operator()(const int input_batches, const int input_height,
                  const int input_width, const int input_channels,
                  const TBitpacked* filter_data, const int filter_height,
                  const int filter_width, const int filter_count,
                  const int stride_rows, const int stride_cols,
                  const int dilation_rows, const int dilation_cols,
                  float* output_data, const int output_height,
                  const int output_width,
                  const float* post_activation_multiplier_data = nullptr,
                  const float* padding_cache = nullptr) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    const int filter_left_offset = ((output_width - 1) * stride_cols +
                                    effective_filter_width - input_width) /
                                   2;
    const int filter_top_offset = ((output_height - 1) * stride_rows +
                                   effective_filter_height - input_height) /
                                  2;

    TFLITE_DCHECK(padding_cache != nullptr);

    // We assume that the input numbers are correct because they
    // are already checked by the bconv functor

    for (int batch = 0; batch < input_batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        // See the `create_cache` function for an explanation of these
        // parameters:
        // How many pixels does the kernel stick out at the
        // left, top, right, bottom
        const int overflow_top = filter_top_offset - out_y * stride_rows;
        const int overflow_bot =
            -overflow_top - input_height + effective_filter_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int overflow_left = filter_left_offset - out_x * stride_cols;
          const int overflow_right =
              -overflow_left - input_width + effective_filter_width;

          if (overflow_left <= 0 && overflow_right <= 0 && overflow_top <= 0 &&
              overflow_bot <= 0) {
            // clang-format off
            // The kernel does not stick out.
            // This output pixel corresponds to a completely 'valid' region
            // of the input and there is no padding: we are entering the inside
            // of the image.
            // We now *skip* from the left to the right edge of the image.
            // We want to find `out_x` such that `overflow_right >= 1`
            // We have
            // overflow_right = out_x * stride - offset + effective_filter_width - width
            // So we want
            // out_x * stride >= width + offset - effective_filter_width + 1
            // So we want `out_x` to be the ceiling of
            // (input_with + offset - effective_filter_width + 1) / stride
            // clang-format on
            int new_out_x = (input_width + filter_left_offset -
                             effective_filter_width + stride_cols - 1) /
                                stride_cols -
                            1;
            // The extra -1 is because the for-loop will increment it again
            if (new_out_x > out_x) out_x = new_out_x;
            continue;
          }
          // See the `create_cache` function for an explanation of these cases
          int case_num;
          int cache_X;
          int cache_Y;
          if (overflow_right <= 0 && overflow_top > 0 && overflow_bot < 0) {
            case_num = 0;
            cache_X = (overflow_left >= 0 ? overflow_left : 0);
            cache_Y = overflow_top;
          } else if (overflow_left < 0 && overflow_right > 0 &&
                     overflow_bot <= 0) {
            case_num = 1;
            cache_X = overflow_right;
            cache_Y = (overflow_top >= 0 ? overflow_top : 0);
          } else if (overflow_left > 0 && overflow_right < 0 &&
                     overflow_top <= 0) {
            case_num = 2;
            cache_X = overflow_left;
            cache_Y = (overflow_bot >= 0 ? overflow_bot : 0);
          } else if (overflow_left <= 0 && overflow_top < 0 &&
                     overflow_bot > 0) {
            case_num = 3;
            cache_X = (overflow_right >= 0 ? overflow_right : 0);
            cache_Y = overflow_bot;
          } else {
            // This cannot happen.
            continue;
          }

          // Cache is out_channels last
          // Output is also out_channels last
          // So we can have a very effective loop here

          // Cache shape is [case, filter_height, filter_width,
          // out_channels]
          int cache_idx = case_num * (effective_filter_height *
                                      effective_filter_width * filter_count) +
                          cache_Y * (effective_filter_width * filter_count) +
                          cache_X * filter_count;
          // Output shape is [batch, height, width, out_channels]
          const int out_idx =
              batch * output_height * output_width * filter_count +
              out_y * output_width * filter_count + out_x * filter_count;

          const float* cache_ptr = &padding_cache[cache_idx];
          float* output_ptr = &output_data[out_idx];

          // Apply pre-computed correction
          // im2col padded the input with 0s which effectively became +1s.
          // The convolution therefore computed
          // out = correct_part + (+1) * (outside_filter_values)
          // So to correct for this we add (-1) * (outside_filter_values)
          for (int out_c = 0; out_c < filter_count; ++out_c) {
            *output_ptr++ += *cache_ptr++;
          }

        }  // out_x
      }    // out_y
    }      // batch
    return;
  }
};

}  // namespace bconv2d
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BCONV2D_PADDING_FUNCTOR_H_
