#ifndef COMPUTE_ENGINE_KERNELS_IM2COL_H_
#define COMPUTE_ENGINE_KERNELS_IM2COL_H_
#include <cstring>

namespace compute_engine {
namespace core {

// this is a naive implementation of im2col algorithm based on the Caffe
// implementation for debugging and testing purposes.
// assumes the input data is stored in "channels_first" format [channels,
// height, width] and the output matrix is stored in a row-major memory layout.
template <class T>
class ReferenceIm2ColFunctorCHW {
 public:
  void operator()(const T* data_im, const int num_channels, const int im_h,
                  const int im_w, const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w, const int stride_h,
                  const int stride_w, const int dilation_h,
                  const int dilation_w, T* data_col) {
    // compute the height and width of the output matrix
    const int output_h =
        (im_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (im_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    const int channel_size = im_h * im_w;
    // iterate through all channels
    for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
      // iterate through kernel cells
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          // iterate through output rows
          for (int output_row = 0; output_row < output_h; ++output_row) {
            if (input_row >= 0 && input_row < im_h) {
              // inside the image
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = 0; output_col < output_w; ++output_col) {
                if (input_col >= 0 && input_col < im_w) {
                  // inside the image
                  *(data_col++) = data_im[input_row * im_w + input_col];
                } else {
                  // outside the left and right boundaries of the image
                  *(data_col++) = 0;
                }
                // compute the next input col considering the width strides
                input_col += stride_w;
              }  // end of loop over output cols
            } else {
              // outside the top and bottom boundaries of image
              for (int output_col = 0; output_col < output_w; ++output_col) {
                *(data_col++) = 0;
              }
            }
            // compute the next input row considering the height strides
            input_row += stride_h;
          }  // end of loop over output rows
        }
      }
      // update the data_im to point to the next channel
      data_im += channel_size;
    }  // end of loop over num_channels
  }
};

// this is a naive implementation of im2col algorithm based on the Caffe
// implementation for debugging and testing purposes.
// assumes the input data is stored in "channels_last" format [height, width,
// channels] and the output matrix is stored in a row-major memory layout.
template <class T>
class ReferenceIm2ColFunctorHWC {
 public:
  void operator()(const T* data_im, const int num_channels, const int im_h,
                  const int im_w, const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w, const int stride_h,
                  const int stride_w, const int dilation_h,
                  const int dilation_w, T* data_col) {
    // compute the height and width of the output matrix
    const int output_h =
        (im_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (im_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // get the size of one T type element in bytes
    const auto data_elem_size_in_bytes = sizeof(T);

    int input_row = 0;
    // iterate through output rows
    for (int output_row = 0; output_row < output_h; ++output_row) {
      int input_col = 0;
      // iterate through output cols
      for (int output_col = 0; output_col < output_w; ++output_col) {
        // iterate through kernel rows
        for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
          // compute the row index of copying element
          int y = input_row - pad_h + kernel_row * dilation_h;
          // compute the address of the yth row in input data
          const T* data_im_at_row_y = data_im + y * im_w * num_channels;
          // iterate through kernel cols
          for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
            // compute the col index of copying element
            int x = input_col - pad_w + kernel_col * dilation_w;
            if (y < 0 || y >= im_h || x < 0 || x >= im_w) {
              // outside image
              memset(data_col, 0, num_channels * data_elem_size_in_bytes);
            } else {
              // inside image -> copy all corresponding channels of element
              // (x,y)
              memcpy(data_col, data_im_at_row_y + x * num_channels,
                     num_channels * data_elem_size_in_bytes);
            }
            data_col += num_channels;
          }  // end of loop over kernel cols
        }    // end of loop over kernel rows
        input_col += stride_w;
      }  // end of loop over output cols
      input_row += stride_h;
    }  // end of loop over output rows
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_IM2COL_H_
