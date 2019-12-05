#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/matrix.h"

using namespace ruy;

template <int LhsCols, int RhsCols>
struct BinaryKernelParams32 {
  const std::uint32_t* lhs_base_ptr;
  const std::uint32_t* rhs_base_ptr;
  float* dst_base_ptr;
  const std::uint32_t* bias;
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
  const std::uint32_t zero_data[LhsCols] = {0};
  std::uint32_t dst_tmp_buf[LhsCols * RhsCols];
};

template <int LhsCols, int RhsCols>
inline void MakeBinaryKernelParams32(
    const PackedMatrix<std::uint32_t>& lhs,
    const PackedMatrix<std::uint32_t>& rhs,
    const BasicSpec<std::int32_t, float>& spec, int start_row, int start_col,
    int end_row, int end_col, Matrix<float>* dst,
    BinaryKernelParams32<LhsCols, RhsCols>* params) {
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
  // params->bias = params->zero_data;
  // if (spec.bias) {
  //   params->bias = spec.bias;
  //   flags |= RUY_ASM_FLAG_HAS_BIAS;
  // }
  params->flags = flags;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(std::uint32_t) * lhs.layout.stride;
  params->rhs_stride = sizeof(std::uint32_t) * rhs.layout.stride;
  params->dst_stride = sizeof(std::uint32_t) * dst->layout.stride;
  params->depth = depth;
  // params->clamp_min = spec.clamp_min;
  // params->clamp_max = spec.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
