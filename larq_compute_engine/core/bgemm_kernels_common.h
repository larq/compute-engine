#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_

#include "larq_compute_engine/core/bconv2d_output_transform.h"
#include "ruy/kernel_common.h"

using namespace ruy;

using compute_engine::core::OutputTransform;

// Our version of `ruy::MulParams`; The original is in `ruy/mul_params.h`.
// We simply use our `OutputTransform` struct.
template <typename tAccumScalar, typename tDstScalar>
struct BinaryMulParams {
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;

  OutputTransform<AccumScalar, DstScalar> output_transform;

  static constexpr LoopStructure kLoopStructure = LoopStructure::kAuto;
  static constexpr LayoutSupport kLayoutSupport = LayoutSupport::kGeneral;
  static constexpr ZeroPointSupport kZeroPointSupport =
      ZeroPointSupport::kGeneral;
  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  // Returns (a reasonable estimate of) the local CPU cache size.
  // See ruy::LocalDataCacheSize() which returns some coarse, sane default for
  // each CPU architecture.
  // This may be overridden, either to provide more accurate/runtime values,
  // or to test with other values to let testcases have more coverage.
  static int local_data_cache_size() { return LocalDataCacheSize(); }
  // Same as local_data_cache_size but for the total data cache size accessible
  // to each CPU core. See ruy::SharedDataCacheSize().
  static int shared_data_cache_size() { return SharedDataCacheSize(); }
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
  float dst_tmp_buf[LhsCols * RhsCols];
};

template <int LhsCols, int RhsCols, typename AccumScalar, typename T>
inline void MakeBinaryKernelParams(
    const PMat<T>& lhs, const PMat<T>& rhs,
    const BinaryMulParams<AccumScalar, float>& spec, int start_row,
    int start_col, int end_row, int end_col, Mat<float>* dst,
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

  params->post_activation_multiplier =
      spec.output_transform.post_activation_multiplier;
  params->post_activation_bias = spec.output_transform.post_activation_bias;
  params->backtransform_add = spec.output_transform.backtransform_add;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(T) * lhs.layout.stride;
  params->rhs_stride = sizeof(T) * rhs.layout.stride;
  params->dst_stride = sizeof(float) * dst->layout.stride;
  params->depth = depth;
  params->clamp_min = spec.output_transform.clamp_min;
  params->clamp_max = spec.output_transform.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

// A specialised template for the case when the LHS and RHS are uint32 bitpacked
// but we're using a kernel designed for uint64 bitpacked inputs.
template <int LhsCols, int RhsCols, typename AccumScalar>
inline void MakeBinaryKernelParams(
    const PackedMatrix<std::uint32_t>& lhs,
    const PackedMatrix<std::uint32_t>& rhs,
    const BinaryMulParams<AccumScalar, float>& spec, int start_row,
    int start_col, int end_row, int end_col, Matrix<float>* dst,
    BinaryKernelParams<LhsCols, RhsCols, std::uint64_t>* params) {
  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = reinterpret_cast<std::uint64_t*>(
      lhs.data + start_row * lhs.layout.stride);
  params->rhs_base_ptr = reinterpret_cast<std::uint64_t*>(
      rhs.data + start_col * rhs.layout.stride);
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;

  params->post_activation_multiplier =
      spec.output_transform.post_activation_multiplier;
  params->post_activation_bias = spec.output_transform.post_activation_bias;
  params->backtransform_add = spec.output_transform.backtransform_add;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(std::uint32_t) * lhs.layout.stride;
  params->rhs_stride = sizeof(std::uint32_t) * rhs.layout.stride;
  params->dst_stride = sizeof(float) * dst->layout.stride;
  // We halve the depth to pretend that the input data is uint64.
  RUY_DCHECK_EQ(depth % 2, 0);
  params->depth = depth / 2;
  params->clamp_min = spec.output_transform.clamp_min;
  params->clamp_max = spec.output_transform.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
