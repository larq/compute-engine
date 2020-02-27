#include "larq_compute_engine/core/bgemm_impl.h"
#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"
#include "utils.h"

using namespace tflite;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template <class T>
inline void im2col(const ConvParams& params, const RuntimeShape& input_shape,
                   const T* input_data, const RuntimeShape& filter_shape,
                   const RuntimeShape& output_shape,
                   const RuntimeShape& im2col_shape, T* im2col_data,
                   RuntimeShape& result_shape, const T** result_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;

  const uint8 zero_byte = 0x00;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  const RuntimeShape* shape = nullptr;
  if (need_dilated_im2col) {
    optimized_ops::DilatedIm2col<T>(params, zero_byte, input_shape, input_data,
                                    filter_shape, output_shape, im2col_data);
    *result_data = im2col_data;
    shape = &im2col_shape;
  } else if (need_im2col) {
    TF_LITE_ASSERT(im2col_data);
    optimized_ops::Im2col<T>(params, filter_height, filter_width, zero_byte,
                             input_shape, input_data, im2col_shape,
                             im2col_data);
    *result_data = im2col_data;
    shape = &im2col_shape;
  } else {
    TF_LITE_ASSERT(!im2col_data);
    *result_data = input_data;
    shape = &input_shape;
  }
  result_shape.ReplaceWith(shape->DimensionsCount(), shape->DimsData());
}

// The inputs post_mutiply and post_add are currently float
// in order to accomodate for batchnorm scales
// Later this might be changed to the int8 system of multipliers+shifts
template <class T, class TBitpacked>
inline void BConv2D(const ConvParams& params, const RuntimeShape& input_shape,
                    const T* input_data, const RuntimeShape& filter_shape,
                    const TBitpacked* packed_filter_data,
                    const float* post_multiply_data, const float* post_add_data,
                    const RuntimeShape& output_shape, T* output_data,
                    const RuntimeShape& im2col_shape, T* im2col_data,
                    bool bitpack_before_im2col, T* padding_buffer,
                    const int pad_value,
                    CpuBackendContext* cpu_backend_context) {
  TF_LITE_ASSERT_EQ(input_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(filter_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(output_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(input_shape.Dims(3), filter_shape.Dims(3));

  gemmlowp::ScopedProfilingLabel label("BConv2D");

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

  // Buffer for bitpacked input data
  static std::vector<TBitpacked> input_data_bp;

  if (bitpack_before_im2col) {
    // Input tensor has this shape which we bitpack along the channels dimension
    //  [batch, input height, input width, channels]
    RuntimeShape packed_input_shape;
    packbits_tensor(input_shape, input_data, packed_input_shape, input_data_bp);

    // Filter tensor was already bitpacked. Only get the new shape
    RuntimeShape packed_filter_shape = packed_shape<TBitpacked>(filter_shape);

    // Get the im2col data buffer
    TBitpacked* packed_im2col_data = reinterpret_cast<TBitpacked*>(im2col_data);

    RuntimeShape result_shape;
    im2col<TBitpacked>(params, packed_input_shape, input_data_bp.data(),
                       packed_filter_shape, output_shape, im2col_shape,
                       packed_im2col_data, result_shape, &rhs_data);

    k = result_shape.Dims(3);
    m = FlatSizeSkipDim(result_shape, 3);
  } else {  // bitpack after im2col

    RuntimeShape result_shape;
    const T* result_data;
    im2col<T>(params, input_shape, input_data, filter_shape, output_shape,
              im2col_shape, im2col_data, result_shape, &result_data);

    // The RHS tensor has this shape which we bitpack along the last dimension
    //  [batch, output_height, output_width, k * bitwidth]
    RuntimeShape packed_input_shape;
    packbits_tensor(result_shape, result_data, packed_input_shape,
                    input_data_bp);
    rhs_data = input_data_bp.data();

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

  cpu_backend_gemm::MatrixParams<T> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;

  // Accumulation type, destination type
  BGemmParams<std::int32_t, T> gemm_params;
  gemm_params.backtransform_add =
      filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
  gemm_params.post_multiply = post_multiply_data;
  gemm_params.post_add = post_add_data;

  // #if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  // TODO: TF lite, on devices which provide optimized BLAS library,
  // uses BLAS instead of the RUY GEMM kernels. For benchmarking we
  // should keep that in mind and also consider developing a
  // BLAS-inspired binary GEMM

  // #endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  BGemm(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, output_data,
        gemm_params, cpu_backend_context);

  if (params.padding_type == PaddingType::kSame && pad_value == 0) {
    using PaddingFunctor =
        ce::core::PaddingFunctor<T, T, ce::core::FilterFormat::OHWI>;

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
      gemmlowp::ScopedProfilingLabel label3("ZeroPaddingCorrection");
      padding_functor(batches, input_height, input_width, input_depth, nullptr,
                      filter_height, filter_width, output_depth, stride_height,
                      stride_width, dilation_height_factor,
                      dilation_width_factor, output_data, output_height,
                      output_width, post_multiply_data, padding_buffer);
    }
  }
}

}  // namespace tflite
}  // namespace compute_engine
