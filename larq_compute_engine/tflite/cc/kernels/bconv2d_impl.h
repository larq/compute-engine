#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"

#include "bgemm_impl.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {

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
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
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

  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  BGemm(lhs_params, filter_data, rhs_params, gemm_input_data,
        dst_params, output_data, gemm_params,
        cpu_backend_context);
}

}  // namespace tflite
}  // namespace compute_engine
