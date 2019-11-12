#ifndef COMPUTE_ENGINE_KERNELS_PADDING_H_
#define COMPUTE_ENGINE_KERNELS_PADDING_H_

#include "larq_compute_engine/cc/utils/types.h"

namespace compute_engine {
namespace core {

namespace ce = compute_engine;

//
// Applies (in-place) corrections for zero-padding
// Assumes that padding type is 'SAME'.
// Currently assumes filter is not bitpacked.
//
template <class Tdata, class Tfilter, FilterFormat filter_format>
class PaddingFunctor {
 public:
  std::size_t get_cache_size(const int filter_height, const int filter_width,
                             const int filter_count, const int dilation_rows,
                             const int dilation_cols) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    return 4 * effective_filter_height * effective_filter_width * filter_count;
  }

  void create_cache(const Tfilter* filter_data, const int filter_height,
                    const int filter_width, const int filter_count,
                    const int input_channels, const int dilation_rows,
                    const int dilation_cols, Tdata* output_cache) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    // Given an (out_x, out_y) for which you want to compute the correction:
    //
    // Let Xl = filter_left_offset - out_x * stride_cols
    // Let Yt = filter_top_offset - out_y * stride_rows
    // Let Xr = -Xl + effecitve_filter_width - input_width
    // Let Yb = -Yt + effecitve_filter_height - input_height
    //
    // Basically:
    //  Xl: how many pixels the kernel sticks out at the left
    //  Yt: how many pixels the kernel sticks out at the top
    //  Xr: how many pixels the kernel sticks out at the right
    //  Yb: how many pixels the kernel sticks out at the bottom
    //
    // The correction that we need to apply depends on (Xl,Yt,Xr,Yb).
    //
    // The correction is a sum over values in the kernel.
    // Now we describe which kernel pixels contribute to this correction.
    //
    // Let efx = dilation_rows * filter_x  //effective filter x
    // Let efy = dilation_cols * filter_y  //effective filter y
    //
    // So fix a (Xl,Yt,Xr,Yb) tuple and loop over filter_x, filter_y
    // If one of the 4 following are satisfied then this
    // filter value is counted for this tuple as correction.
    // (T)        efy <  Yt     this pixel sticks out at Top
    // (B)  efh - efy <= Yb     this pixel sticks out at Bottom
    // (L)        efx <  Xl     this pixel sticks out at Left
    // (R)  efw - efx <= Xr     this pixel sticks out at Right
    //
    // So we sum this over all kernel pixels that stick out somewhere
    // to obtain our correction value.
    //
    // So we could save this value for every (Xl,Yt,Xr,Yb) tuple.
    // However (T) and (B) can not be both true at the same time,
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
    //
    //          Xl      Xr      Yt      Yb
    // Case 0   *       <= 0    >  0    <  0
    // Case 1   <  0    >  0    *       <= 0
    // Case 2   >  0    <  0    <= 0    *
    // Case 3   <= 0    *       <  0    >  0
    //
    // For pixels in the top middle, we have Xl < 0,
    // but they all share the same correction, namely the
    // correction for Xl = 0.
    // The same happens for every * in the table.
    //
    // Therefore, if there is a * then we should take x=relu(x)
    //

    for (int Y = 0; Y < effective_filter_height; ++Y) {
      for (int X = 0; X < effective_filter_width; ++X) {
        for (int out_c = 0; out_c < filter_count; ++out_c) {
          Tdata corrections[4] = {0, 0, 0, 0};

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              // Sum over input channels
              Tdata cur_correction = Tdata(0);
              for (int in_c = 0; in_c < input_channels; ++in_c) {
                int filter_idx;
                if (filter_format == FilterFormat::HWIO) {
                  // filter_data has shape
                  // [height, width, in_channels, out_channels]
                  filter_idx = filter_y * (filter_width * input_channels *
                                           filter_count) +
                               filter_x * (input_channels * filter_count) +
                               in_c * filter_count + out_c;
                } else {
                  // filter_data has shape
                  // [out_channels, height, width, in_channels]
                  filter_idx =
                      out_c * (filter_height * filter_width * input_channels) +
                      filter_y * (filter_width * input_channels) +
                      filter_x * input_channels + in_c;
                }

                cur_correction += filter_data[filter_idx];
              }

              const int efx = dilation_cols * filter_x;
              const int efy = dilation_rows * filter_y;
              // Case 0: Xl = X; Yt = Y
              if (efy < Y || efx < X) {
                corrections[0] += cur_correction;
              }
              // Case 1: Xr = X; Yt = Y
              if (efy < Y || (effective_filter_width - efx) <= X) {
                corrections[1] += cur_correction;
              }
              // Case 2: Xl = X; Yb = Y
              if ((effective_filter_height - efy) <= Y || efx < X) {
                corrections[2] += cur_correction;
              }
              // Case 3: Xr = X; Yb = Y
              if ((effective_filter_height - efy) <= Y ||
                  (effective_filter_width - efx) <= X) {
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
                Y * (effective_filter_width * filter_count) + X * filter_count +
                out_c;
            output_cache[output_idx] = corrections[direction];
          }
        }
      }
    }
    return;
  }

  void operator()(const int input_batches, const int input_height,
                  const int input_width, const int input_channels,
                  const Tdata* padding_cache, const int filter_height,
                  const int filter_width, const int filter_count,
                  const int stride_rows, const int stride_cols,
                  const int dilation_rows, const int dilation_cols,
                  Tdata* output_data, const int output_height,
                  const int output_width) {
    const int effective_filter_width = (filter_width - 1) * dilation_cols + 1;
    const int effective_filter_height = (filter_height - 1) * dilation_rows + 1;

    const int filter_left_offset = ((output_width - 1) * stride_cols +
                                    effective_filter_width - input_width) /
                                   2;
    const int filter_top_offset = ((output_height - 1) * stride_rows +
                                   effective_filter_height - input_height) /
                                  2;

    // We assume that the input numbers are correct because they
    // are already checked by the bconv functor

    for (int batch = 0; batch < input_batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        // See the `create_cache` function for an explanation of these
        // parameters:
        // How many pixels does the kernel stick out at the
        // left, top, right, bottom
        const int Yt = filter_top_offset - out_y * stride_rows;
        const int Yb = -Yt - input_height + effective_filter_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int Xl = filter_left_offset - out_x * stride_cols;
          const int Xr = -Xl - input_width + effective_filter_width;

          if (Xl <= 0 && Xr <= 0 && Yt <= 0 && Yb <= 0) {
            // clang-format off
            // The kernel does not stick out.
            // This output pixel corresponds to a completely 'valid' region
            // of the input and there is no padding: we are entering the inside
            // of the image.
            // We now *skip* from the left to the right edge of the image.
            // We want to find `out_x` such that `Xr >= 1`
            // We have
            // Xr = out_x * stride - offset + effective_filter_width - width
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
          if (Xr <= 0 && Yt > 0 && Yb < 0) {
            case_num = 0;
            cache_X = (Xl >= 0 ? Xl : 0);
            cache_Y = Yt;
          } else if (Xl < 0 && Xr > 0 && Yb <= 0) {
            case_num = 1;
            cache_X = Xr;
            cache_Y = (Yt >= 0 ? Yt : 0);
          } else if (Xl > 0 && Xr < 0 && Yt <= 0) {
            case_num = 2;
            cache_X = Xl;
            cache_Y = (Yb >= 0 ? Yb : 0);
          } else if (Xl <= 0 && Yt < 0 && Yb > 0) {
            case_num = 3;
            cache_X = (Xr >= 0 ? Xr : 0);
            cache_Y = Yb;
          } else {
            // This can not happen.
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

          const Tdata* cache_ptr = &padding_cache[cache_idx];
          Tdata* output_ptr = &output_data[out_idx];

          // Apply pre-computed correction
          // im2col padded the input with 0s which effectively became -1s.
          // The convolution therefore computed
          // out = correct_part + (-1) * (outside_filter_values)
          // So to correct for this we add (+1) * (outside_filter_values)
          for (int out_c = 0; out_c < filter_count; ++out_c) {
            *output_ptr++ += *cache_ptr++;
          }
        }  // out_x
      }    // out_y
    }      // batch
    return;
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_PADDING_H_
