#include "bgemm_impl.h"
#include "larq_compute_engine/cc/core/packbits.h"
#include "larq_compute_engine/cc/core/padding_functor.h"
#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template <class T, class TBitpacked>
inline void BConv2D(const ConvParams& params, const RuntimeShape& input_shape,
                    const T* input_data, const RuntimeShape& filter_shape,
                    const T* filter_data, const RuntimeShape& bias_shape,
                    const T* bias_data, const RuntimeShape& output_shape,
                    T* output_data, const RuntimeShape& im2col_shape,
                    T* im2col_data, T* padding_buffer,
                    CpuBackendContext* cpu_backend_context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const T output_activation_min = params.float_activation_min;
  const T output_activation_max = params.float_activation_max;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_EQ(input_shape.Dims(3), filter_shape.Dims(3));

  gemmlowp::ScopedProfilingLabel label("BConv2D");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const TBitpacked* gemm_input_data = nullptr;
  int gemm_input_rows = 0, gemm_input_cols = 0;

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int in_channels = input_shape.Dims(3);
  const int gemm_depth_dimension = filter_height * filter_width * in_channels;

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  constexpr auto bitwidth = std::numeric_limits<TBitpacked>::digits;
  bool bitpack_first = (in_channels % bitwidth == 0);

  static std::vector<TBitpacked> rhs_data_bp;

  if (bitpack_first) {
    // RHS input tensor has shape
    // [batch, input height, input width, channels]
    // and we now view it as a matrix of shape
    // [batch * input height * input width, channels]
    // and bitpack it along the channels dimension.
    const int rhs_rows = FlatSizeSkipDim(input_shape, 3);
    const int rhs_cols = in_channels;
    size_t rhs_rows_bp = 0, rhs_cols_bp = 0;
    size_t rhs_bitpadding = 0;
    {
      gemmlowp::ScopedProfilingLabel label2("PackbitRHS");
      const T* rhs_data = input_data;
      ce::core::packbits_matrix(rhs_data, rhs_rows, rhs_cols, rhs_data_bp,
                                rhs_rows_bp, rhs_cols_bp, rhs_bitpadding,
                                ce::core::Axis::RowWise);
    }
    TFLITE_DCHECK_EQ(rhs_bitpadding, 0);

    const TBitpacked* packed_input_data = rhs_data_bp.data();
    const int packed_in_channels = in_channels / bitwidth;

    RuntimeShape packed_input_shape(input_shape);
    RuntimeShape packed_filter_shape(filter_shape);
    packed_input_shape.SetDim(3, packed_in_channels);
    packed_filter_shape.SetDim(3, packed_in_channels);

    // With bitpacking first, we could in principle allocate a *smaller*
    // im2col_data buffer, but we will keep the bigger buffer fow now,
    // and just change the shape.
    TBitpacked* packed_im2col_data = reinterpret_cast<TBitpacked*>(im2col_data);
    // im2col_shape:
    // [batch, output height, output width, channels_in * filter_h * filter_w]
    RuntimeShape packed_im2col_shape(im2col_shape);
    packed_im2col_shape.SetDim(3, im2col_shape.Dims(3) / bitwidth);

    TFLITE_DCHECK_EQ(packed_im2col_shape.Dims(3),
                     gemm_depth_dimension / bitwidth);

    const RuntimeShape* gemm_input_shape = nullptr;
    if (need_dilated_im2col) {
      optimized_ops::DilatedIm2col<TBitpacked>(
          params, float_zero_byte, packed_input_shape, packed_input_data,
          packed_filter_shape, output_shape, packed_im2col_data);
      gemm_input_data = packed_im2col_data;
      gemm_input_shape = &packed_im2col_shape;
    } else if (need_im2col) {
      TFLITE_DCHECK(im2col_data);
      optimized_ops::Im2col<TBitpacked>(params, filter_height, filter_width,
                                        float_zero_byte, packed_input_shape,
                                        packed_input_data, packed_im2col_shape,
                                        packed_im2col_data);
      gemm_input_data = packed_im2col_data;
      gemm_input_shape = &packed_im2col_shape;
    } else {
      TFLITE_DCHECK(!im2col_data);
      gemm_input_data = packed_input_data;
      gemm_input_shape = &packed_input_shape;
    }
    gemm_input_rows = gemm_input_shape->Dims(3);
    gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  } else {  // im2col first

    const T* rhs_data = nullptr;
    const RuntimeShape* gemm_input_shape = nullptr;
    if (need_dilated_im2col) {
      optimized_ops::DilatedIm2col(params, float_zero_byte, input_shape,
                                   input_data, filter_shape, output_shape,
                                   im2col_data);
      rhs_data = im2col_data;
      gemm_input_shape = &im2col_shape;
    } else if (need_im2col) {
      TFLITE_DCHECK(im2col_data);
      optimized_ops::Im2col(params, filter_height, filter_width,
                            float_zero_byte, input_shape, input_data,
                            im2col_shape, im2col_data);
      rhs_data = im2col_data;
      gemm_input_shape = &im2col_shape;
    } else {
      TFLITE_DCHECK(!im2col_data);
      rhs_data = input_data;
      gemm_input_shape = &input_shape;
    }

    const int gemm_input_dims = gemm_input_shape->DimensionsCount();
    int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
    int n = output_shape.Dims(3);
    int k = gemm_input_shape->Dims(gemm_input_dims - 1);

    // row-wise bitpacking of RHS (m, k) -> (m, k / bitwidth)
    const int rhs_rows = m;
    const int rhs_cols = k;
    size_t rhs_rows_bp = 0, rhs_cols_bp = 0;
    size_t rhs_bitpadding = 0;
    // TODO: pre-allocate the 'rhs_data_bp' buffer
    // 'packbits_matrix' function calls the 'resize' method of the container
    // in case that bitpadding is required. Therefore, in order to pre-allocate
    // bitpacked buffer, we need to redesign the packbits_matrix and move
    // computing the size of the bitpacked buffer outside the 'packbits_matrix'
    // function. For now we just define the bitpacking buffer as static.
    {
      gemmlowp::ScopedProfilingLabel label2("PackbitRHS");
      ce::core::packbits_matrix(rhs_data, rhs_rows, rhs_cols, rhs_data_bp,
                                rhs_rows_bp, rhs_cols_bp, rhs_bitpadding,
                                ce::core::Axis::RowWise);
    }
    gemm_input_data = rhs_data_bp.data();

    // Since the layout is colmajor, we flip the rows and cols
    gemm_input_rows = rhs_cols_bp;
    gemm_input_cols = rhs_rows_bp;
  }

  // `filter_data` is a matrix with dimensions (n, k)
  // `gemm_input_data` is a matrix with dimensions (m, k)
  // `output_data` is a matrix with dimensions (n, m)
  // We would like to compute the following matrix-matrix mulitplicaiton:
  //
  // 'output_data' (m, n) = 'gemm_input_data' (m, k) * 'filter_data' (k, n)
  //
  // We use the 'filter_data' as LHS and assume is stored with RowMajor layout
  // -> (n, k). We use 'gemm_input_data' as RHS and assume is stored in
  // ColMajor layout -> (k, m). If we assume the 'output_data' is stored in
  // ColMajor layout (m, n), then it can be calculated as following:
  //
  // 'output_data' (m, n) = 'filter_data' (n, k) x 'gemm_input_data' (m, k)
  //
  // where `x` is row-wise dotprod of the LHS and RHS.
  // We perform the bitpacking compression along the k-dimension so the
  // following is acutally computed in BGEMM: 'output_data' (m, n) =
  // 'filter_data' (n, k / bitwitdh) x 'gemm_input_data' (m, k / bitwidth)

  // LHS is the filter values with shape
  // [output channels, height, width, input channels]
  // and we now view it as a matrix of shape
  // [output channels, height * width * input_channels]
  // and bitpack it along the last dimension
  const auto* lhs_data = filter_data;

  // row-wise bitpacking of LHS (n, k) -> (n, k / bitwidth)
  const int lhs_rows = filter_shape.Dims(0);
  const int lhs_cols = gemm_depth_dimension;
  // TODO: pre-allocate the 'lhs_data_bp' buffer in prepare
  // 'packbits_matrix' function calls the 'resize' method of the container
  // in case that bitpadding is required. Therefore, in order to pre-allocate
  // bitpacked buffer, we need to redesign the packbits_matrix and move
  // computing the size of the bitpacked buffer outside the 'packbits_matrix'
  // function. For now we just define the bitpacking buffer as static.
  static std::vector<TBitpacked> lhs_data_bp;
  size_t lhs_rows_bp = 0, lhs_cols_bp = 0;
  size_t lhs_bitpadding = 0;
  {
    gemmlowp::ScopedProfilingLabel label1("PackbitLHS");
    ce::core::packbits_matrix(lhs_data, lhs_rows, lhs_cols, lhs_data_bp,
                              lhs_rows_bp, lhs_cols_bp, lhs_bitpadding,
                              ce::core::Axis::RowWise);
  }

  // LHS (n, k/bitwidth) -> RowMajor -> (n, k/bitwidth)
  // RHS (m, k/bitwidth) -> ColMajor -> (k/bitwidth, m)
  // DST (n, m) -> ColMajor -> (m, n)
  cpu_backend_gemm::MatrixParams<TBitpacked> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = lhs_rows_bp;  // n
  lhs_params.cols = lhs_cols_bp;  // k/bitwidth

  cpu_backend_gemm::MatrixParams<TBitpacked> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = gemm_input_rows;  // k/bitwidth
  rhs_params.cols = gemm_input_cols;  // m

  cpu_backend_gemm::MatrixParams<T> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows_bp;
  dst_params.cols = gemm_input_cols;

  // TODO: Currently GemmParams is not used the same way
  // as it is used in the TF Lite codebase. Here, we abuse the
  // 'multiplier_exponent' which is used only for non-floating-point
  // cases, to pass the bitpadding correction value (int) to BGemm
  cpu_backend_gemm::GemmParams<TBitpacked, T> gemm_params;
  gemm_params.multiplier_exponent = lhs_bitpadding;
  // gemm_params.bias = bias_data;
  // gemm_params.clamp_min = output_activation_min;
  // gemm_params.clamp_max = output_activation_max;

  // #if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  // TODO: TF lite, on devices which provide optimized BLAS library,
  // uses BLAS instead of the RUY GEMM kernels. For benchmarking we
  // should keep that in mind and also consider developing a
  // BLAS-inspired binary GEMM

  // #endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  BGemm(lhs_params, lhs_data_bp.data(), rhs_params, gemm_input_data, dst_params,
        output_data, gemm_params, cpu_backend_context);

  if (params.padding_type == PaddingType::kSame) {
    using PaddingFunctor =
        ce::core::PaddingFunctor<T, T, ce::core::FilterFormat::OHWI>;

    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int input_width = input_shape.Dims(2);
    const int input_height = input_shape.Dims(1);
    const int output_depth = output_shape.Dims(3);
    const int output_width = output_shape.Dims(2);
    const int output_height = output_shape.Dims(1);

    PaddingFunctor padding_functor;
    {
      gemmlowp::ScopedProfilingLabel label3("ZeroPaddingCorrection");
      padding_functor(batches, input_height, input_width, input_depth,
                      filter_data, filter_height, filter_width, output_depth,
                      stride_height, stride_width, dilation_height_factor,
                      dilation_width_factor, output_data, output_height,
                      output_width, padding_buffer);
    }
  }
}

}  // namespace tflite
}  // namespace compute_engine
