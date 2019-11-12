#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"

#include "bgemm_impl.h"
#include "larq_compute_engine/cc/core/packbits.h"

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
                    T* im2col_data, CpuBackendContext* cpu_backend_context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const T output_activation_min = params.float_activation_min;
  const T output_activation_max = params.float_activation_max;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;
  (void)im2col_shape;
  gemmlowp::ScopedProfilingLabel label("BConv2D");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const T* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  if (need_dilated_im2col) {
    optimized_ops::DilatedIm2col(params, float_zero_byte, input_shape,
                                 input_data, filter_shape, output_shape,
                                 im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::Im2col(params, filter_height, filter_width, float_zero_byte,
                          input_shape, input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  // `filter_data` is a matrix with dimensions (n, k)
  // `gemm_input_data` is a matrix with dimensions (m, k)
  // `output_data` is a matrix with dimensions (n, m)
  // We would like to compute the following matrix-matrix mulitplicaiton:
  //
  // 'output_data' (m, n) = 'gemm_input_data' (m, k) * 'filter_data' (k, n)
  //
  // We use the 'filter_data' as LHS and assume is stored with RowMajor layout
  // -> (n, k). We use 'gemm_input_data' as RHS and assume is stored in ColMajor
  // layout -> (k, m). If we assume the 'output_data' is stored in ColMajor
  // layout (m, n), then it can be calculated as following:
  //
  // 'output_data' (m, n) = 'filter_data' (n, k) x 'gemm_input_data' (m, k)
  //
  // where `x` is row-wise dotprod of the LHS and RHS.
  // We perform the bitpacking compression along the k-dimension so the
  // following is acutally computed in BGEMM: 'output_data' (m, n) =
  // 'filter_data' (n, k / bitwitdh) x 'gemm_input_data' (m, k / bitwidth)

  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

  // LHS is the filter values
  const auto* lhs_data = filter_data;

  // RHS is the input values
  const auto* rhs_data = gemm_input_data;

  // row-wise bitpacking of LHS (n, k) -> (n, k / bitwidth)
  const int lhs_rows = n;
  const int lhs_cols = k;
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
  // row-wise bitpacking of RHS (m, k) -> (m, k / bitwidth)
  const int rhs_rows = m;
  const int rhs_cols = k;
  // TODO: pre-allocate the 'rhs_data_bp' buffer
  // 'packbits_matrix' function calls the 'resize' method of the container
  // in case that bitpadding is required. Therefore, in order to pre-allocate
  // bitpacked buffer, we need to redesign the packbits_matrix and move
  // computing the size of the bitpacked buffer outside the 'packbits_matrix'
  // function. For now we just define the bitpacking buffer as static.
  static std::vector<TBitpacked> rhs_data_bp;
  size_t rhs_rows_bp = 0, rhs_cols_bp = 0;
  size_t rhs_bitpadding = 0;
  {
    gemmlowp::ScopedProfilingLabel label2("PackbitRHS");
    ce::core::packbits_matrix(rhs_data, rhs_rows, rhs_cols, rhs_data_bp,
                              rhs_rows_bp, rhs_cols_bp, rhs_bitpadding,
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
  // Since the layout is colmajor, we flip the rows and cols
  rhs_params.rows = rhs_cols_bp;  // k/bitwidth
  rhs_params.cols = rhs_rows_bp;  // m

  cpu_backend_gemm::MatrixParams<T> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;

  // TODO: Currently GemmParmas is not used the same way
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

  BGemm(lhs_params, lhs_data_bp.data(), rhs_params, rhs_data_bp.data(),
        dst_params, output_data, gemm_params, cpu_backend_context);

  if (params.padding_type == PaddingType::kSame) {
    using PaddingFunctor =
        ce::core::ReferencePaddingFunctor<T, T, ce::core::FilterFormat::OHWI>;

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
                      stride_height, stride_width, output_data, output_height,
                      output_width);
    }
  }
}

}  // namespace tflite
}  // namespace compute_engine
