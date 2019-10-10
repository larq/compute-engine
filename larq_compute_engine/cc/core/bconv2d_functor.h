#ifndef COMPUTE_ENGINE_KERNELS_BCONV2D_H_
#define COMPUTE_ENGINE_KERNELS_BCONV2D_H_

#include <iostream>

#include "larq_compute_engine/cc/core/fused_bgemm_functor.h"

namespace compute_engine {
namespace core {

namespace ce = compute_engine;

// We don't want to allocate a buffer to hold all the patches if the size is
// going to be extremely large, so break it into chunks if it's bigger than
// a limit. Each chunk will be processed serially, so we can refill the
// buffer for the next chunk and reuse it, keeping maximum memory size down.
// In this case, we've picked 16 megabytes as a reasonable limit for Android and
// other platforms using Eigen, and 1MB for Apple devices, from experimentation.
#if defined(__APPLE__) && defined(IS_MOBILE_PLATFORM)
const size_t kMaxChunkSize = (1 * 1024 * 1024);
#else
const size_t kMaxChunkSize = (16 * 1024 * 1024);
#endif

// Implements convolution as a two stage process, first packing the patches of
// the input image into columns (im2col) and then running GEMM to produce the
// final result.
template <class T1, class T2, class T3, class TGemmFunctor>
class Im2ColBConvFunctor {
 public:
  void operator()(const T1* input_data, int input_batches, int input_height,
                  int input_width, int input_depth, const T2* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride_rows, int stride_cols, int padding,
                  T3* output_data, int output_height, int output_width) {
    if ((input_batches <= 0) || (input_width <= 0) || (input_height <= 0) ||
        (input_depth <= 0)) {
      std::cout << "Conv2D was called with bad input dimensions: "
                << input_batches << ", " << input_height << ", " << input_width
                << ", " << input_depth;
      return;
    }
    if ((filter_width <= 0) || (filter_height <= 0) || (filter_count <= 0)) {
      std::cout << "Conv2D was called with bad filter dimensions: "
                << filter_width << ", " << filter_height << ", "
                << filter_count;
      return;
    }
    if ((output_width <= 0) || (output_height <= 0)) {
      std::cout << "Conv2D was called with bad output width or height: "
                << output_width << ", " << output_height;
      return;
    }

    // We can just use a GEMM if the im2col is the identity operator, e.g., if
    // the kernel is 1x1 or the input data and filter have same height/width.
    if (filter_height == 1 && filter_width == 1 && stride_rows == 1 &&
        stride_cols == 1) {
      // The kernel is 1x1.
      const int m = input_batches * input_height * input_width;
      const int n = filter_count;
      const int k = input_depth;
      const int lda = k;
      const int ldb = filter_count;
      const int ldc = filter_count;
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, input_data, lda, filter_data, ldb, output_data,
                   ldc);
      return;
    } else if (filter_height == input_height && filter_width == input_width &&
               padding == 1) {
      // The input data and filter have the same height/width.
      const int m = input_batches;
      const int n = filter_count;
      const int k = input_height * input_width * input_depth;
      const int lda = k;
      const int ldb = filter_count;
      const int ldc = filter_count;
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, input_data, lda, filter_data, ldb, output_data,
                   ldc);
      return;
    }

    // These calculations define how the patches will be positioned within the
    // input image. The actual definitions are quite complex, and rely on the
    // previously-calculated output size.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == 1) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // The im2col buffer has # of patches rows, and # of filters cols.
    // It's laid out like this, in row major order in memory:
    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    // Each patch row contains a filter_width x filter_height patch of the
    // input, with the depth channel as the most contiguous in memory, followed
    // by the width, then the height. This is the standard memory order in the
    // image world if it helps to visualize it.
    const int filter_value_count = filter_width * filter_height * input_depth;
    // OP_REQUIRES(context, (filter_value_count * sizeof(T1)) <= kMaxChunkSize,
    //             errors::InvalidArgument("Im2Col patch too large for
    //             buffer"));
    typedef long long int64;
    const int64 patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    const int64 chunk_value_count =
        (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);

    std::vector<T1> im2col_buffer_resource(chunk_value_count);
    T1* im2col_buffer = im2col_buffer_resource.data();

    const int64 patch_count = (input_batches * output_height * output_width);
    const int64 chunk_count =
        (patch_count + (patches_per_chunk - 1)) / patches_per_chunk;
    for (int64 chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64 patch_index_start = chunk_index * patches_per_chunk;
      const int64 patch_index_end =
          std::min(patch_index_start + patches_per_chunk, patch_count);
      for (int64 patch_index = patch_index_start; patch_index < patch_index_end;
           ++patch_index) {
        const int64 batch = patch_index / (output_height * output_width);
        const int64 out_y = (patch_index / output_width) % output_height;
        const int64 out_x = patch_index % output_width;
        const T1* input_batch_start =
            input_data + (batch * input_height * input_width * input_depth);
        const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
        const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
        const int patch_index_within_chunk = patch_index % patches_per_chunk;
        T1* im2col_patch_start =
            im2col_buffer + (patch_index_within_chunk * filter_value_count);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y;
          T1* im2col_row_start =
              im2col_patch_start + (filter_y * filter_width * input_depth);
          // If we're off the top or the bottom of the input, fill the
          // whole row with zeroes.
          if ((in_y < 0) || (in_y >= input_height)) {
            T1* im2col_row_end =
                im2col_row_start + (filter_width * input_depth);
            std::fill(im2col_row_start, im2col_row_end, T1(0));
          } else {
            // What we're doing here is trying to copy and fill the im2col
            // buffer as efficiently as possible, using functions to set or
            // duplicate values en masse. We know we don't have to worry about
            // vertical edges because we dealt with that case above, so we
            // just need to handle filters that overlap the left or right
            // edges. Here's what that looks like:
            //
            // < left_zero_count > < center_copy_count > < right_zero_count >
            // +------------------+---------------------+--------------------+
            // |     (filter)     |       (image)       |      (filter)      |
            // +------------------+---------------------+--------------------+
            // in_x_origin        0                 input_width       in_x_end
            //
            // In reality it's unlikely that a filter patch will be wider
            // than an input, but this shows all the edge cases.
            // We use std::fill() to set the left and right sections to zeroes
            // and std::copy() to copy over the input data for the center.
            const int in_x_end = in_x_origin + filter_width;
            const int left_zero_count = std::max(0, 0 - in_x_origin);
            const int right_zero_count = std::max(0, in_x_end - input_width);
            const int center_copy_count =
                filter_width - (left_zero_count + right_zero_count);
            if (left_zero_count > 0) {
              T1* im2col_left_start = im2col_row_start;
              T1* im2col_left_end =
                  im2col_left_start + (left_zero_count * input_depth);
              std::fill(im2col_left_start, im2col_left_end, T1(0));
            }
            if (center_copy_count > 0) {
              const T1* input_row_start =
                  input_batch_start + (in_y * input_width * input_depth) +
                  (std::max(0, in_x_origin) * input_depth);
              const T1* input_row_end =
                  input_row_start + (center_copy_count * input_depth);
              T1* im2col_center_start =
                  im2col_row_start + (left_zero_count * input_depth);
              std::copy(input_row_start, input_row_end, im2col_center_start);
            }
            if (right_zero_count > 0) {
              T1* im2col_right_start =
                  im2col_row_start +
                  ((left_zero_count + center_copy_count) * input_depth);
              T1* im2col_right_end =
                  im2col_right_start + (right_zero_count * input_depth);
              std::fill(im2col_right_start, im2col_right_end, T1(0));
            }
          }
        }
      }
      // Now we've assembled a set of image patches into a matrix, apply a
      // GEMM matrix multiply of the patches as rows, times the filter
      // weights in columns, to get partial results in the output matrix.
      const int how_many_patches = patch_index_end - patch_index_start;
      const int m = how_many_patches;
      const int n = filter_count;
      const int k = filter_value_count;
      const int lda = filter_value_count;
      const int ldb = filter_count;
      const int ldc = filter_count;
      T3* chunk_output_data = output_data + (patch_index_start * filter_count);
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, im2col_buffer, lda, filter_data, ldb,
                   chunk_output_data, ldc);
    }
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_BCONV2D_H_
