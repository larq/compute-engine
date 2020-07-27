#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_

#include <cassert>

#include "larq_compute_engine/core/bgemm_impl.h"
#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/packbits_utils.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace wip_ops {
template <typename T>
inline void ExtractPatchIntoBufferColumn(
    const RuntimeShape& input_shape, int w, int h, int b, int kheight,
    int kwidth, int stride_width, int stride_height, int pad_width,
    int pad_height, int in_width, int in_height, int in_depth,
    int kwidth_times_indepth, int inwidth_times_indepth,
    int single_buffer_length, int buffer_id, const T* in_data,
    T* conv_buffer_data, uint8 zero_byte) {
  ruy::profiler::ScopeLabel label("ExtractPatchIntoBufferColumn");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  // This chunk of code reshapes all the inputs corresponding to
  // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
  const int ih_ungated_start = h * stride_height - pad_height;
  const int ih_ungated_end = (ih_ungated_start + kheight);
  const int ih_end = std::min(ih_ungated_end, in_height);
  const int iw_ungated_start = w * stride_width - pad_width;
  const int iw_ungated_end = (iw_ungated_start + kwidth);
  const int iw_end = std::min(iw_ungated_end, in_width);
  // If the patch is off the edge of the input image, skip writing those rows
  // and columns from the patch into the output array.
  const int h_offset = std::max(0, -ih_ungated_start);
  const int w_offset = std::max(0, -iw_ungated_start);
  const int ih_start = std::max(0, ih_ungated_start);
  const int iw_start = std::max(0, iw_ungated_start);
  const int single_row_num =
      std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
  const int output_row_offset = (buffer_id * single_buffer_length);
  int out_offset =
      output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
  int in_offset = Offset(input_shape, b, ih_start, iw_start, 0);

  // Express all of the calculations as padding around the input patch.
  const int top_padding = h_offset;
  const int bottom_padding = (ih_ungated_end - ih_end);
  const int left_padding = w_offset;
  const int right_padding = (iw_ungated_end - iw_end);
  assert(single_row_num ==
         ((kwidth - (left_padding + right_padding)) * in_depth));

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0) {
    const int top_row_elements = (top_padding * kwidth * in_depth);
    memset(conv_buffer_data + output_row_offset, zero_byte,
           (top_row_elements * sizeof(T)));
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if (left_padding > 0) {
    if (right_padding > 0) {
      for (int ih = ih_start; ih < ih_end; ++ih) {
        const int left_start = (out_offset - (left_padding * in_depth));
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + left_start, zero_byte,
               (left_padding * in_depth * sizeof(T)));
        memcpy(conv_buffer_data + out_offset, in_data + in_offset,
               single_row_num * sizeof(T));
        memset(conv_buffer_data + right_start, zero_byte,
               (right_padding * in_depth * sizeof(T)));
        out_offset += kwidth_times_indepth;
        in_offset += inwidth_times_indepth;
      }
    } else {
      for (int ih = ih_start; ih < ih_end; ++ih) {
        const int left_start = (out_offset - (left_padding * in_depth));
        memset(conv_buffer_data + left_start, zero_byte,
               (left_padding * in_depth * sizeof(T)));
        memcpy(conv_buffer_data + out_offset, in_data + in_offset,
               single_row_num * sizeof(T));
        out_offset += kwidth_times_indepth;
        in_offset += inwidth_times_indepth;
      }
    }
  } else {
    if (right_padding > 0) {
      for (int ih = ih_start; ih < ih_end; ++ih) {
        const int right_start = (out_offset + single_row_num);
        memcpy(conv_buffer_data + out_offset, in_data + in_offset,
               single_row_num * sizeof(T));
        memset(conv_buffer_data + right_start, zero_byte,
               (right_padding * in_depth * sizeof(T)));
        out_offset += kwidth_times_indepth;
        in_offset += inwidth_times_indepth;
      }
    } else {
      for (int ih = ih_start; ih < ih_end; ++ih) {
        memcpy(conv_buffer_data + out_offset, in_data + in_offset,
               single_row_num * sizeof(T));
        out_offset += kwidth_times_indepth;
        in_offset += inwidth_times_indepth;
      }
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0) {
    const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
    const int bottom_start =
        output_row_offset +
        ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
    memset(conv_buffer_data + bottom_start, zero_byte,
           (bottom_row_elements * sizeof(T)));
  }
}

template <typename T>
void Im2col(const ConvParams& params, int kheight, int kwidth, uint8 zero_byte,
            const RuntimeShape& input_shape, const T* input_data,
            const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Im2col");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  const int kwidth_times_indepth = kwidth * input_depth;
  const int inwidth_times_indepth = input_width * input_depth;

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        ExtractPatchIntoBufferColumn(
            input_shape, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            kwidth_times_indepth, inwidth_times_indepth, output_depth,
            buffer_id, input_data, output_data, zero_byte);
        ++buffer_id;
      }
    }
  }
}

template <typename T>
void Im2col(const ConvParams& params, int kheight, int kwidth,
            const int32_t* input_offsets, const int input_offsets_size,
            const RuntimeShape& input_shape, const T* input_data,
            const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Im2col");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  TFLITE_DCHECK_EQ(batches, input_offsets_size);
  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  const int kwidth_times_indepth = kwidth * input_depth;
  const int inwidth_times_indepth = input_width * input_depth;

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b) {
    uint8_t zero_byte = static_cast<uint8_t>(input_offsets[b]);
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        ExtractPatchIntoBufferColumn(
            input_shape, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            kwidth_times_indepth, inwidth_times_indepth, output_depth,
            buffer_id, input_data, output_data, zero_byte);
        ++buffer_id;
      }
    }
  }
}
}  // namespace wip_ops
}  // namespace tflite

using namespace tflite;
namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template <class T>
inline void im2col(const ConvParams& params, const RuntimeShape& input_shape,
                   const T* input_data, const RuntimeShape& filter_shape,
                   const RuntimeShape& output_shape,
                   const RuntimeShape& im2col_shape, T* im2col_data,
                   RuntimeShape& result_shape, const T** result_data,
                   const std::int32_t zero_point) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;

  const std::uint8_t zero_byte = (std::uint8_t)zero_point;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  const RuntimeShape* shape = nullptr;
  if (need_dilated_im2col) {
    TF_LITE_ASSERT(im2col_data);
    optimized_ops::DilatedIm2col<T>(params, zero_byte, input_shape, input_data,
                                    filter_shape, output_shape, im2col_data);
    *result_data = im2col_data;
    shape = &im2col_shape;
  } else if (need_im2col) {
    TF_LITE_ASSERT(im2col_data);
    wip_ops::Im2col<T>(params, filter_height, filter_width, zero_byte,
                       input_shape, input_data, im2col_shape, im2col_data);
    *result_data = im2col_data;
    shape = &im2col_shape;
  } else {
    TF_LITE_ASSERT(!im2col_data);
    *result_data = input_data;
    shape = &input_shape;
  }
  result_shape.ReplaceWith(shape->DimensionsCount(), shape->DimsData());
}

// Get the post_activation_multiplier out of the OutputTransform struct
// Required for the padding functor
template <typename AccumScalar, typename DstScalar>
const float* GetPostActivationMultiplier(
    const OutputTransform<AccumScalar, DstScalar>& output_transform) {
  return nullptr;
}
template <typename AccumScalar>
const float* GetPostActivationMultiplier(
    const OutputTransform<AccumScalar, float>& output_transform) {
  return output_transform.post_activation_multiplier;
}

template <typename SrcScalar, typename TBitpacked, typename AccumScalar,
          typename DstScalar>
inline void BConv2D(
    const ConvParams& params, const RuntimeShape& input_shape,
    const SrcScalar* input_data, TBitpacked* packed_input_data,
    const RuntimeShape& filter_shape, const TBitpacked* packed_filter_data,
    const OutputTransform<AccumScalar, DstScalar>& output_transform,
    const RuntimeShape& output_shape, DstScalar* output_data,
    const RuntimeShape& im2col_shape, SrcScalar* im2col_data,
    bool bitpack_before_im2col, const float* padding_buffer,
    const int pad_value, const bool read_bitpacked_input,
    CpuBackendContext* cpu_backend_context) {
  TF_LITE_ASSERT_EQ(input_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(filter_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(output_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT(read_bitpacked_input ||
                 input_shape.Dims(3) == filter_shape.Dims(3));
  TF_LITE_ASSERT(!read_bitpacked_input || bitpack_before_im2col);

  ruy::profiler::ScopeLabel label("BConv2D");

  //                   m
  //              ___________
  //             |1 7        |
  //             |2 8        |
  //             |3 9        |
  //            k|. . inputs |
  //             |. .        |
  //      k      |___________|
  //   ________
  //  |123..   |
  //  |789..   |
  //  |        |    k = filter_height * filter_width * channels_in / bitwidth
  // n| filter |    m = output_height * output_width
  //  |        |    n = output_channels
  //  |________|
  //
  // Storage order is shown in the matrices
  //
  // bitpack_before_im2col
  //    inputs and filters are packed along the `channels_in` dimension.
  // else
  //    inputs and filters are packed along the k-dimension
  //
  const TBitpacked* lhs_data = packed_filter_data;
  const TBitpacked* rhs_data = nullptr;

  int n = filter_shape.Dims(0);
  int m = 0;
  int k = 0;

  if (bitpack_before_im2col) {
    // The filter tensor was already bitpacked. Only get the new shape.
    RuntimeShape packed_filter_shape =
        ce::core::packed_shape<TBitpacked>(filter_shape);

    // Get the im2col data buffer.
    TBitpacked* packed_im2col_data = reinterpret_cast<TBitpacked*>(im2col_data);

    // We're already bitpacked, so im2col `zero_byte` is 0
    RuntimeShape result_shape;

    RuntimeShape packed_input_shape = input_shape;
    const TBitpacked* im2col_input_data;
    if (read_bitpacked_input) {
      im2col_input_data = reinterpret_cast<const TBitpacked*>(input_data);
    } else {
      // The input tensor has this shape which we bitpack along the channels
      // dimension [batch, input height, input width, channels].
      ruy::profiler::ScopeLabel label("Bitpack activations (before im2col)");
      ce::core::packbits_tensor<ce::core::BitpackOrder::Optimized>(
          input_shape, input_data, params.input_offset, packed_input_shape,
          packed_input_data);
      im2col_input_data = packed_input_data;
    }
    im2col<TBitpacked>(params, packed_input_shape, im2col_input_data,
                       packed_filter_shape, output_shape, im2col_shape,
                       packed_im2col_data, result_shape, &rhs_data, 0);

    k = result_shape.Dims(3);
    m = FlatSizeSkipDim(result_shape, 3);
  } else {  // Bitpack after im2col.
    // We need to pad with the correct zero_byte
    RuntimeShape result_shape;
    const SrcScalar* result_data;
    im2col<SrcScalar>(params, input_shape, input_data, filter_shape,
                      output_shape, im2col_shape, im2col_data, result_shape,
                      &result_data, params.input_offset);

    // The RHS tensor has this shape which we bitpack along the last dimension
    //  [batch, output_height, output_width, k * bitwidth]
    RuntimeShape packed_input_shape;
    {
      ruy::profiler::ScopeLabel label("Bitpack activations (after im2col)");
      ce::core::packbits_tensor<ce::core::BitpackOrder::Optimized>(
          result_shape, result_data, params.input_offset, packed_input_shape,
          packed_input_data);
    }
    rhs_data = packed_input_data;

    k = packed_input_shape.Dims(3);
    m = FlatSizeSkipDim(packed_input_shape, 3);
  }

  // LHS (n, k/bitwidth) -> RowMajor -> (n, k/bitwidth)
  // RHS (m, k/bitwidth) -> ColMajor -> (k/bitwidth, m)
  // DST (n, m) -> ColMajor -> (m, n)
  cpu_backend_gemm::MatrixParams<TBitpacked> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;

  cpu_backend_gemm::MatrixParams<TBitpacked> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;

  cpu_backend_gemm::MatrixParams<DstScalar> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;

  // #if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  // TODO: TF lite, on devices which provide optimized BLAS library,
  // uses BLAS instead of the RUY GEMM kernels. For benchmarking we
  // should keep that in mind and also consider developing a
  // BLAS-inspired binary GEMM

  // #endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  BGemm(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, output_data,
        output_transform, cpu_backend_context);

  if (params.padding_type == PaddingType::kSame && pad_value == 0) {
    using PaddingFunctor =
        ce::core::PaddingFunctor<DstScalar, float, float, float,
                                 ce::core::FilterFormat::OHWI>;

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int input_width = input_shape.Dims(2);
    const int input_height = input_shape.Dims(1);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);
    const int output_width = output_shape.Dims(2);
    const int output_height = output_shape.Dims(1);

    PaddingFunctor padding_functor;
    {
      ruy::profiler::ScopeLabel label3("ZeroPaddingCorrection");
      padding_functor(
          batches, input_height, input_width, input_depth, nullptr,
          filter_height, filter_width, output_depth, stride_height,
          stride_width, dilation_height_factor, dilation_width_factor,
          output_data, output_height, output_width,
          GetPostActivationMultiplier(output_transform), padding_buffer);
    }
  }
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_
