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
  const AccumScalar* fused_multiply = nullptr;
  const AccumScalar* fused_add = nullptr;
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
  const AccumScalar* fused_multiply = nullptr;
  const AccumScalar* fused_add = nullptr;
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
  const std::int32_t* fused_multiply;
  const std::int32_t* fused_add;
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
  std::uint32_t clamp_min;
  std::uint32_t clamp_max;
  std::uint8_t flags;
  const T zero_data[LhsCols] = {0};
  T dst_tmp_buf[LhsCols * RhsCols];
};

template <int LhsCols, int RhsCols, class T>
inline void MakeBinaryKernelParams(
    const PackedMatrix<T>& lhs, const PackedMatrix<T>& rhs,
    const BinaryBasicSpec<std::int32_t, float>& spec,
    int start_row, int start_col, int end_row, int end_col, Matrix<float>* dst,
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
  params->fused_multiply = spec.fused_multiply;
  params->fused_add = spec.fused_add;
  if (spec.fused_multiply && spec.fused_add) {
    flags |= RUY_ASM_FLAG_HAS_BIAS;
  }
  params->flags = flags;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(T) * lhs.layout.stride;
  params->rhs_stride = sizeof(T) * rhs.layout.stride;
  params->dst_stride = sizeof(float) * dst->layout.stride;
  params->depth = depth;
  // params->clamp_min = spec.clamp_min;
  // params->clamp_max = spec.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
