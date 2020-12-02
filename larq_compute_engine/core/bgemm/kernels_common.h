#ifndef COMPUTE_ENGINE_CORE_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_ENGINE_CORE_BGEMM_KERNELS_COMMON_H_

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/kernel_common.h"

namespace compute_engine {
namespace core {
namespace bgemm {

using namespace ruy;
using ruy::Order;  // To fix Windows build

using namespace bitpacking;
using bconv2d::OutputTransform;

// Our version of `ruy::MulParams`; The original is in `ruy/mul_params.h`.
// We simply use our `OutputTransform` struct.
template <typename tAccumScalar, typename tDstScalar>
struct BinaryMulParams {
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;

  OutputTransform<DstScalar> output_transform;

  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
};

// A specialisation for the writing-bitpacked-output case, where the C++ LHS
// kernel layout must have `bitpacking_bitwidth` columns (so that that many
// channels can be bitpacked at once).
template <typename tAccumScalar>
struct BinaryMulParams<tAccumScalar, TBitpacked> {
  using AccumScalar = tAccumScalar;
  using DstScalar = TBitpacked;

  OutputTransform<DstScalar> output_transform;

  using StandardCppKernelLhsLayout =
      FixedKernelLayout<Order::kColMajor, 1, bitpacking_bitwidth>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
};

template <typename DstScalar, int LhsCols, int RhsCols>
struct BinaryKernelParams {
  const TBitpacked* lhs_base_ptr;
  const TBitpacked* rhs_base_ptr;
  DstScalar* dst_base_ptr;
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
  OutputTransform<DstScalar> output_transform;
  DstScalar dst_tmp_buf[LhsCols * RhsCols];  // Used for float or int8 output
};

template <typename AccumScalar, typename DstScalar, int LhsCols, int RhsCols>
inline void MakeBinaryKernelParams(
    const PMat<TBitpacked>& lhs, const PMat<TBitpacked>& rhs, int start_row,
    int start_col, int end_row, int end_col, Mat<DstScalar>* dst,
    const BinaryMulParams<AccumScalar, DstScalar>& mul_params,
    BinaryKernelParams<DstScalar, LhsCols, RhsCols>* params) {
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);
  if (std::is_same<DstScalar, TBitpacked>::value) {
    RUY_DCHECK_EQ(start_row % 8, 0);
    RUY_DCHECK_EQ(end_row % 8, 0);
  }

  params->lhs_base_ptr = lhs.data + start_row * lhs.layout.stride;
  params->rhs_base_ptr = rhs.data + start_col * rhs.layout.stride;
  if (std::is_same<DstScalar, TBitpacked>::value) {
    // When writing bitpacked output with multiple threads, the start/end row
    // will not necessarily be aligned to a TBitpacked boundary (though they are
    // guaranteed to be aligned to a byte boundary). For example, start_row
    // could be 8, in which case we'd be writing into the second byte of each
    // TBitpacked value. Hence the need for char pointer arithmetic.
    params->dst_base_ptr =
        (DstScalar*)((char*)(dst->data.get() +
                             start_col * GetBitpackedSize(dst->layout.stride)) +
                     start_row / 8);
  } else {
    params->dst_base_ptr =
        dst->data.get() + start_col * dst->layout.stride + start_row;
  }
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;
  params->lhs_stride = sizeof(TBitpacked) * lhs.layout.stride;
  params->rhs_stride = sizeof(TBitpacked) * rhs.layout.stride;
  params->dst_stride =
      sizeof(DstScalar) * (std::is_same<DstScalar, TBitpacked>::value
                               ? GetBitpackedSize(dst->layout.stride)
                               : dst->layout.stride);
  params->depth = lhs.layout.rows;
  // This is a four word copy, but that's okay.
  params->output_transform = mul_params.output_transform;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

}  // namespace bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_KERNELS_COMMON_H_
