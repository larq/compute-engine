#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/kernel_common.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace ruy;

using tflite::cpu_backend_gemm::QuantizationFlavor;

// Our version of `cpu_backend_gemm::GemmParams`
// Original is in `lite/kernels/cpu_backend_gemm_params.h`
// Modifications:
// - bias changes to multiply + add
// - clamp_min, clamp_max type changed from DstScalar to AccumScalar
// - (later) 8-bit quantization related stuff
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor =
              std::is_floating_point<DstScalar>::value
                  ? QuantizationFlavor::kFloatingPoint
                  : QuantizationFlavor::kIntegerWithUniformMultiplier>
struct BGemmParams {
  AccumScalar multiplier_fixedpoint = 0;
  int multiplier_exponent = 0;
  int32_t backtransform_add = 0;
  // post_mutiply and post_activation_bias are currently float
  // in order to accomodate for batchnorm scales
  // Later this might be changed to the int8 system of multipliers+shifts
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;
  AccumScalar clamp_min = std::numeric_limits<AccumScalar>::lowest();
  AccumScalar clamp_max = std::numeric_limits<AccumScalar>::max();
};

//
// Our version of `ruy::BasicSpec`
// Original is in `lite/experimental/ruy/spec.h`
// Modifications:
// - bias changes to multiply + add
// - clamp_min, clamp_max types changed from DstScalar to AccumScalar
// - (later) 8-bit quantization related stuff

template <typename tAccumScalar, typename tDstScalar>
struct BinaryBasicSpec {
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;
  AccumScalar multiplier_fixedpoint = 0;
  int multiplier_exponent = 0;
  int32_t backtransform_add = 0;
  // post_mutiply and post_activation_bias are currently float
  // in order to accomodate for batchnorm scales
  // Later this might be changed to the int8 system of multipliers+shifts
  const float* post_activation_multiplier = nullptr;
  const float* post_activation_bias = nullptr;
  AccumScalar clamp_min = std::numeric_limits<AccumScalar>::lowest();
  AccumScalar clamp_max = std::numeric_limits<AccumScalar>::max();

  // This is identical to `ruy::BasicSpec`
  static constexpr LoopStructure kLoopStructure = LoopStructure::kAuto;
  static constexpr LayoutSupport kLayoutSupport = LayoutSupport::kGeneral;
  static constexpr ZeroPointSupport kZeroPointSupport =
      ZeroPointSupport::kGeneral;
  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  static int cache_friendly_traversal_threshold() { return 32 * 1024; }
};

template <int LhsCols, int RhsCols, class T>
struct BinaryKernelParams {
  const T* lhs_base_ptr;
  const T* rhs_base_ptr;
  float* dst_base_ptr;
  // post_mutiply and post_activation_bias are currently float
  // in order to accomodate for batchnorm scales
  // Later this might be changed to the int8 system of multipliers+shifts
  const float* post_activation_multiplier;
  const float* post_activation_bias;
  std::int32_t start_row;
  std::int32_t start_col;
  std::int32_t last_row;
  std::int32_t last_col;
  std::int32_t dst_rows;
  std::int32_t dst_cols;
  std::int32_t lhs_stride;
  std::int32_t rhs_stride;
  std::int32_t dst_stride;
  std::int32_t depth;
  std::int32_t clamp_min;
  std::int32_t clamp_max;
  std::int32_t backtransform_add;
  std::uint8_t flags;
  const T zero_data[LhsCols] = {0};
  T dst_tmp_buf[LhsCols * RhsCols];
};

template <int LhsCols, int RhsCols, class T>
inline void MakeBinaryKernelParams(
    const PackedMatrix<T>& lhs, const PackedMatrix<T>& rhs,
    const BinaryBasicSpec<std::int32_t, float>& spec, int start_row,
    int start_col, int end_row, int end_col, Matrix<float>* dst,
    BinaryKernelParams<LhsCols, RhsCols, T>* params) {
  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = lhs.data + start_row * lhs.layout.stride;
  params->rhs_base_ptr = rhs.data + start_col * rhs.layout.stride;
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;

  std::uint8_t flags = 0;
  params->post_activation_multiplier = spec.post_activation_multiplier;
  params->post_activation_bias = spec.post_activation_bias;
  if (spec.post_activation_multiplier && spec.post_activation_bias) {
    flags |= RUY_ASM_FLAG_HAS_BIAS;
  }
  params->backtransform_add = spec.backtransform_add;
  params->flags = flags;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(T) * lhs.layout.stride;
  params->rhs_stride = sizeof(T) * rhs.layout.stride;
  params->dst_stride = sizeof(float) * dst->layout.stride;
  params->depth = depth;
  params->clamp_min = spec.clamp_min;
  params->clamp_max = spec.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
