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
  int operator()(int input_batches, int input_height, int input_width,
                  int input_channels, const Tfilter* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride_rows, int stride_cols, Tdata* output_data,
                  int output_height, int output_width) {
    // We assume that the input numbers are correct because they
    // are already checked by the bconv functor

    if (stride_rows != 1 || stride_cols != 1) {
      return 1;
    }
    if (output_height != input_height || output_width != input_width) {
      return 2;
    }

    // For even-sized kernels, there is an extra pixel of padding at the bottom
    // and the right i.e. the "center pixel" is the top-left one of the 4
    // possible centers. So we split the padding numbers into one for each side
    int pad_x_low = (filter_width - 1) / 2;
    int pad_x_high = filter_width / 2;
    int pad_y_low = (filter_height - 1) / 2;
    int pad_y_high = filter_height / 2;

    // When the filter indices are {0,1,..,s-1}
    // Then the "center pixel" has index (s-1)/2
    //
    // So output index i (centered at input index i)
    // gets input from {i - (s-1)/2, ..., i + s/2}
    // both endpoints included
    //

    for (int batch = 0; batch < input_batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        int in_y_min = out_y - pad_y_low;
        int in_y_max = out_y + pad_y_high;
        if (in_y_min >= 0 && in_y_max < input_height) continue;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          int in_x_min = out_x - pad_x_low;
          int in_x_max = out_x + pad_x_high;
          if (in_x_min >= 0 && in_x_max < input_width) continue;

          for (int out_c = 0; out_c < filter_count; ++out_c) {
            Tdata correction = Tdata(0);

            // TODO: The correction factors that are computed in these loops
            // can be pre-computed.
            for (int f_y = 0; f_y < filter_height; ++f_y) {
              int in_y = out_y - pad_y_low + f_y;
              for (int f_x = 0; f_x < filter_width; ++f_x) {
                int in_x = out_x - pad_x_low + f_x;

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

    return 0;
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_PADDING_H_
