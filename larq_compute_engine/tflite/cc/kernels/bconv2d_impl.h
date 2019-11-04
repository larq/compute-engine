#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"

#include "larq_compute_engine/cc/core/packbits.h"
#include "bgemm_impl.h"

using namespace tflite;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template<class TBitpacked>
inline void BConv2D(const ConvParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& filter_shape,
                    const float* filter_data, const RuntimeShape& bias_shape,
                    const float* bias_data, const RuntimeShape& output_shape,
                    float* output_data, const RuntimeShape& im2col_shape,
                    float* im2col_data, CpuBackendContext* cpu_backend_context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;
  (void)im2col_shape;
  gemmlowp::ScopedProfilingLabel label("BConv2D");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  if (need_dilated_im2col) {
    optimized_ops::DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                                 filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
                          input_data, im2col_shape, im2col_data);
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
  // We use the 'filter_data' as LHS and assume is stored with RowMajor layout (n, k)
  // We use 'gemm_input_data' as RHS and assume is stored in ColMajor layout (k, m)
  // If we assume the 'output_data' is stored in ColMajor layout (m, n), then it can be calculated
  // as following:
  // 'output_data' (n, m) = 'filter_data' (n, k) * 'gemm_input_data' (m, k)
  //
  // We perform the bitpacking compression along the k-dimension so the following
  // is acutally computed in BGEMM:
  // 'output_data' (m, n) = 'filter_data' (n, k / bitwitdh) * 'gemm_input_data' (k / bitwidth, m)
  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

  // LHS is the filter values
  const auto* lhs_data = filter_data;

  // RHS is the input values
  const auto* rhs_data = gemm_input_data;

  // rowwise bitpacking of LHS (n, k) -> (n, k / bitwidth)
  const int lhs_rows = n;
  const int lhs_cols = k;
  // TODO: pre-allocate the 'lhs_data_bp' buffer
  std::vector<TBitpacked> lhs_data_bp;
  size_t lhs_rows_bp = 0, lhs_cols_bp = 0;
  size_t lhs_bitpadding = 0;
  ce::core::packbits_matrix(lhs_data, lhs_rows, lhs_cols, lhs_data_bp, lhs_rows_bp,
                            lhs_cols_bp, lhs_bitpadding, ce::core::Axis::RowWise);

  cpu_backend_gemm::MatrixParams<TBitpacked> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = lhs_rows_bp;
  lhs_params.cols = lhs_cols_bp;

  // rowwise bitpacking of RHS (m, k) -> (m, k / bitwidth)
  const int rhs_rows = m;
  const int rhs_cols = k;
  // TODO: pre-allocate the 'rhs_data_bp' buffer
  std::vector<TBitpacked> rhs_data_bp;
  size_t rhs_rows_bp = 0, rhs_cols_bp = 0;
  size_t rhs_bitpadding = 0;
  ce::core::packbits_matrix(rhs_data, rhs_rows, rhs_cols, rhs_data_bp,
                            rhs_rows_bp,
                            rhs_cols_bp, rhs_bitpadding, ce::core::Axis::RowWise);

  cpu_backend_gemm::MatrixParams<TBitpacked> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  // Since the layout is colmajor, we flip the rows and cols
  rhs_params.rows = rhs_cols_bp;
  rhs_params.cols = rhs_rows_bp;

  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  // Since the layout is colmajor, we flip the rows and cols
  dst_params.rows = n;
  dst_params.cols = m;

  // TODO: Currently GemmParmas is not used in BGEMM
  cpu_backend_gemm::GemmParams<TBitpacked, float> gemm_params;
  // gemm_params.bias = bias_data;
  // gemm_params.clamp_min = output_activation_min;
  // gemm_params.clamp_max = output_activation_max;

  BGemm(lhs_params, lhs_data_bp.data(), rhs_params, rhs_data_bp.data(),
        dst_params, output_data, gemm_params,
        cpu_backend_context);
}

}  // namespace tflite
}  // namespace compute_engine
