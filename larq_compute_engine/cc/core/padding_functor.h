#ifndef COMPUTE_ENGINE_KERNELS_PADDING_H_
#define COMPUTE_ENGINE_KERNELS_PADDING_H_

namespace compute_engine {
namespace core {

namespace ce = compute_engine;

//
// Applies (in-place) corrections for zero-padding
// Assumes that padding type is 'VALID'.
// Currently assumes filter is not bitpacked.
//
// Reference implementation
//
template <class Tdata, class Tfilter>
class PaddingFunctor {
 public:
  void operator()(int input_batches, int input_height, int input_width,
                  int input_channels, const Tfilter* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride_rows, int stride_cols, Tdata* output_data,
                  int output_height, int output_width) {

    int filter_left_offset =
        ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
    int filter_top_offset =
        ((output_height - 1) * stride_rows + filter_height - input_height) / 2;

    // We assume that the input numbers are correct because they
    // are already checked by the bconv functor

    for (int batch = 0; batch < input_batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        int in_y_min = out_y * stride_rows - filter_top_offset;
        int in_y_max = in_y_min + filter_height - 1;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          int in_x_min = out_x * stride_cols - filter_left_offset;
          int in_x_max = in_x_min + filter_width - 1;

          // Skip if this output pixel is in the 'valid' region with no padding
          // TODO: Skip to the correct out_x value at once
          if (in_x_min >= 0 && in_x_max < input_width && in_y_min >= 0 &&
              in_y_max < input_height)
            continue;

          for (int out_c = 0; out_c < filter_count; ++out_c) {
            Tdata correction = Tdata(0);

            // TODO: The correction factors that are computed in these loops
            // can be pre-computed in tf lite.
            for (int f_y = 0; f_y < filter_height; ++f_y) {
              int in_y = in_y_min + f_y;
              for (int f_x = 0; f_x < filter_width; ++f_x) {
                int in_x = in_x_min + f_x;

                // Check if this filter pixel is 'inside' the input image
                if (in_x >= 0 && in_x < input_width && in_y >= 0 &&
                    in_y < input_height)
                  continue;

                for (int in_c = 0; in_c < input_channels; ++in_c) {
                  // filter_data currently has shape
                  // [height, width, in_channels, out_channels]
                  int f_idx =
                      f_y * (filter_width * input_channels * filter_count) +
                      f_x * (input_channels * filter_count) +
                      in_c * filter_count + out_c;

                  correction += filter_data[f_idx];
                }
              }
            }

            // Apply correction
            // During im2col, the image was filled with 0's which effectively
            // became -1's. So the convolution has added -1 * filter_values So
            // we correct by adding filter_values

            // Output shape is [batch, height, width, out_channels]
            int out_idx = batch * output_height * output_width * filter_count +
                          out_y * output_width * filter_count +
                          out_x * filter_count + out_c;
            output_data[out_idx] += correction;
          }
        }
      }
    }
    return;
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_PADDING_H_
