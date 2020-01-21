#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/matrix.h"

using namespace ruy;

template <int LhsCols, int RhsCols, class T>
struct BinaryKernelParams {
  const T* lhs_base_ptr;
  const T* rhs_base_ptr;
  float* dst_base_ptr;
  const std::int32_t* mul_bias;
  const std::int32_t* add_bias;
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
    const BasicSpec<std::int32_t, float>& spec, int start_row, int start_col,
    int end_row, int end_col, Matrix<float>* dst,
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
  if (spec.bias) {
    params->mul_bias = spec.bias;
    params->add_bias = spec.bias + dst->layout.rows;
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
