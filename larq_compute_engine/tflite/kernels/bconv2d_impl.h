#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_

#include "larq_compute_engine/core/bgemm_impl.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

using ce::core::TBitpacked;

inline void im2col(const ConvParams& params, const RuntimeShape& input_shape,
                   const TBitpacked* input_data,
                   const RuntimeShape& filter_shape,
                   const RuntimeShape& output_shape,
                   const RuntimeShape& im2col_shape, TBitpacked* im2col_data,
                   RuntimeShape& result_shape, const TBitpacked** result_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;

  const std::uint8_t zero_byte = 0;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  const RuntimeShape* shape = nullptr;
  if (need_dilated_im2col) {
    TF_LITE_ASSERT(im2col_data);
    optimized_ops::DilatedIm2col<TBitpacked>(params, zero_byte, input_shape,
                                             input_data, filter_shape,
                                             output_shape, im2col_data);
    *result_data = im2col_data;
    shape = &im2col_shape;
  } else if (need_im2col) {
    TF_LITE_ASSERT(im2col_data);
    optimized_ops::Im2col<TBitpacked>(params, filter_height, filter_width,
                                      zero_byte, input_shape, input_data,
                                      im2col_shape, im2col_data);
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

template <typename AccumScalar, typename DstScalar>
inline void BConv2D(
    const ConvParams& params, const RuntimeShape& input_shape,
    const TBitpacked* input_data, const RuntimeShape& filter_shape,
    const TBitpacked* packed_filter_data,
    const OutputTransform<AccumScalar, DstScalar>& output_transform,
    const RuntimeShape& output_shape, DstScalar* output_data,
    const RuntimeShape& im2col_shape, TBitpacked* im2col_data,
    const float* padding_buffer, const int pad_value,
    CpuBackendContext* cpu_backend_context) {
  TF_LITE_ASSERT_EQ(input_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(filter_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(output_shape.DimensionsCount(), 4);

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
  // inputs and filters are packed along the `channels_in` dimension.
  //
  const TBitpacked* lhs_data = packed_filter_data;
  const TBitpacked* rhs_data = nullptr;

  int n = filter_shape.Dims(0);
  int m = 0;
  int k = 0;

  RuntimeShape result_shape;
  im2col(params, input_shape, input_data, filter_shape, output_shape,
         im2col_shape, im2col_data, result_shape, &rhs_data);

  k = result_shape.Dims(3);
  m = FlatSizeSkipDim(result_shape, 3);

  // LHS (n, k/bitwidth) -> RowMajor -> (n, k/bitwidth)
  // RHS (m, k/bitwidth) -> ColMajor -> (k/bitwidth, m)
  // DST (n, m) -> ColMajor -> (m, n)
  cpu_backend_gemm::MatrixParams<TBitpacked> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;
  // The LHS is the weights, which are static, so caching is prefered.
  lhs_params.cache_policy = cpu_backend_gemm::CachePolicy::kAlwaysCache;

  cpu_backend_gemm::MatrixParams<TBitpacked> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;
  // The RHS is the input activations, which change every inference, so there's
  // no advantage from caching.
  rhs_params.cache_policy = cpu_backend_gemm::CachePolicy::kNeverCache;

  cpu_backend_gemm::MatrixParams<DstScalar> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;

  BGemm(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, output_data,
        output_transform, cpu_backend_context);

  if (params.padding_type == PaddingType::kSame && pad_value == 0) {
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

    ce::core::PaddingFunctor padding_functor;
    {
      ruy::profiler::ScopeLabel label3("ZeroPaddingCorrection");
      padding_functor(
          batches, input_height, input_width, input_depth, nullptr,
          filter_height, filter_width, output_depth, stride_height,
          stride_width, dilation_height_factor, dilation_width_factor,
          reinterpret_cast<float*>(output_data), output_height, output_width,
          GetPostActivationMultiplier(output_transform), padding_buffer);
    }
  }
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BCONV_2D_IMPL_H_
