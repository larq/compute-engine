#ifndef COMPUTE_ENGINE_KERNELS_PADDING_H_
#define COMPUTE_ENGINE_KERNELS_PADDING_H_

namespace compute_engine {
namespace core {

enum class FilterFormat { HWIO, OHWI };

namespace ce = compute_engine;

//
// Applies (in-place) corrections for zero-padding
// Assumes that padding type is 'SAME'.
// Currently assumes filter is not bitpacked.
//
// Reference implementation
//
template <class Tdata, class Tfilter, FilterFormat filter_format>
class ReferencePaddingFunctor {
 public:
  void operator()(const int input_batches, const int input_height,
                  const int input_width, const int input_channels,
                  const Tfilter* filter_data, const int filter_height,
                  const int filter_width, const int filter_count,
                  const int stride_rows, const int stride_cols,
                  Tdata* output_data, const int output_height,
                  const int output_width) {
    const int filter_left_offset =
        ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
    const int filter_top_offset =
        ((output_height - 1) * stride_rows + filter_height - input_height) / 2;

    // We assume that the input numbers are correct because they
    // are already checked by the bconv functor

    for (int batch = 0; batch < input_batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_start = out_y * stride_rows - filter_top_offset;
        const int in_y_end = in_y_start + filter_height - 1;  // inclusive
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_start = out_x * stride_cols - filter_left_offset;
          const int in_x_end = in_x_start + filter_width - 1;  // inclusive

          if (in_x_start >= 0 && in_x_end < input_width && in_y_start >= 0 &&
              in_y_end < input_height) {
            // This output pixel corresponds to a completely 'valid' region
            // of the input and there is no padding: we are entering the inside
            // of the image.
            // Therefore we now *skip* from the left to the right edge of the
            // image We want to find `out_x` such that `in_x_end >= input_width`
            // i.e.
            // out_x * stride_cols >= input_with + filter_left_offset -
            // filter_width + 1 So we want it to be the ceiling of (input_with +
            // filter_left_offset - filter_width + 1) / stride_cols
            out_x = (input_width + filter_left_offset - filter_width + 1 +
                     stride_cols - 1) /
                        stride_cols -
                    1;  // -1 because the for-loop will increment it again
            continue;
          }

          for (int out_c = 0; out_c < filter_count; ++out_c) {
            Tdata correction = Tdata(0);

            // TODO: The correction factors that are computed in these loops
            // can be pre-computed in tf lite.
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_start + filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_start + filter_x;

                // Check if this filter pixel is 'inside' the input image
                if (in_x >= 0 && in_x < input_width && in_y >= 0 &&
                    in_y < input_height)
                  continue;

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
                    filter_idx = out_c * (filter_height * filter_width *
                                          input_channels) +
                                 filter_y * (filter_width * input_channels) +
                                 filter_x * input_channels + in_c;
                  }

                  correction += filter_data[filter_idx];
                }
              }
            }

            // Apply correction
            // im2col padded the input with 0s which effectively became -1s.
            // The convolution therefore computed
            // out = correct_part + (-1) * (outside_filter_values)
            // So to correct fot this we add (+1) * (outside_filter_values)

            // Output shape is [batch, height, width, out_channels]
            const int out_idx =
                batch * output_height * output_width * filter_count +
                out_y * output_width * filter_count + out_x * filter_count +
                out_c;
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
