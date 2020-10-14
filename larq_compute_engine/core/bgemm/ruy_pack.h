/*
 * Ruy performs 'packing' of the LHS and RHS matrices before the GEMM operation,
 * which involves re-arranging the matrices in memory to enable linear memory
 * access patterns for the optimised assembly GEMM kernels. For more detail on
 * the motivation for packing, see the the gemmlowp docs:
 * https://github.com/google/gemmlowp/blob/master/doc/packing.md.
 *
 * Packing is part of the inference 'critical path' because although the packed
 * weights matrix can be cached, the packed input activations cannot be. Ruy
 * therefore includes optimised packing routines. However, these are specific to
 * the datatypes (float and/or int8) and kernel-layouts used in the built-in Ruy
 * kernels, which don't match those of our optimised LCE kernels. This means
 * that if we rely on the Ruy packing code we end up falling back to the slow
 * reference implementation that copies each element one-by-one.
 *
 * In our optimised LCE kernels, the LHS matrix is the weights and the RHS is
 * the input activations. As the packed weights can be computed once then
 * cached, we only care about optimising for packing the RHS. In all our
 * optimised kernels, the RHS has a 4x4 column-major kernel layout; we therefore
 * extend the Ruy packing with an optimised (but portable, C++) packing routine
 * for 4x4 column-major kernel layouts.
 */

#ifndef COMPUTE_ENGINE_CORE_BGEMM_RUY_PACK_H_
#define COMPUTE_ENGINE_CORE_BGEMM_RUY_PACK_H_

#include "larq_compute_engine/core/types.h"
#include "ruy/pack_common.h"

namespace compute_engine {
namespace core {
namespace bgemm {

using namespace ruy;
template <Path ThePath, typename FixedKernelLayout, Order SrcOrder>
struct LceRuyPackImpl : PackImpl<ThePath, FixedKernelLayout, TBitpacked,
                                 TBitpacked, TBitpacked, SrcOrder> {};

template <Path ThePath>
struct LceRuyPackImpl<ThePath, FixedKernelLayout<Order::kColMajor, 4, 4>,
                      Order::kColMajor> {
  // This method is derived Ruy's 16x4 column-major int8 packing routine:
  // https://github.com/google/ruy/blob/c9f5f9cecde3d6314df6e7d91517356bf07135eb/ruy/pack_arm.h#L122-L197
  static void Run(Tuning, const Mat<TBitpacked>& src_matrix,
                  PMat<TBitpacked>* packed_matrix, int start_col, int end_col) {
    profiler::ScopeLabel label("Pack (ColMajor, 4x4)");

    // Likewise, Ruy supports arbitrary zero points, but we only use true-zero.
    RUY_DCHECK_EQ(src_matrix.zero_point, 0);

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);

    TBitpacked zerobuf[4];
    memset(zerobuf, 0, sizeof(zerobuf));

    const int src_stride = src_matrix.layout.stride;

    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      const TBitpacked* src_ptr0 =
          src_matrix.data.get() + src_stride * block_col;
      const TBitpacked* src_ptr1 = src_ptr0 + src_stride;
      const TBitpacked* src_ptr2 = src_ptr1 + src_stride;
      const TBitpacked* src_ptr3 = src_ptr2 + src_stride;
      int src_inc0 = 4;
      int src_inc1 = 4;
      int src_inc2 = 4;
      int src_inc3 = 4;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        src_ptr3 = zerobuf;
        src_inc3 = 0;
      }
      TBitpacked* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      constexpr std::size_t packed_stride = 4;
      constexpr std::size_t packed_stride_bytes = sizeof(TBitpacked) * 4;
      for (int block_row = 0; block_row < src_matrix.layout.rows - 3;
           block_row += 4) {
        memcpy(packed_ptr, src_ptr0, packed_stride_bytes);
        src_ptr0 += src_inc0;
        memcpy(packed_ptr + packed_stride, src_ptr1, packed_stride_bytes);
        src_ptr1 += src_inc1;
        memcpy(packed_ptr + 2 * packed_stride, src_ptr2, packed_stride_bytes);
        src_ptr2 += src_inc2;
        memcpy(packed_ptr + 3 * packed_stride, src_ptr3, packed_stride_bytes);
        src_ptr3 += src_inc3;
        packed_ptr += 4 * packed_stride;
      }
      if (src_matrix.layout.rows % 4 != 0) {
        const std::size_t non_zero_rows = src_matrix.layout.rows % 4;
        const std::size_t non_zero_size = sizeof(TBitpacked) * non_zero_rows;
        const std::size_t zero_size = packed_stride_bytes - non_zero_size;
        memcpy(packed_ptr, src_ptr0, non_zero_size);
        memset(packed_ptr + non_zero_rows, 0, zero_size);
        packed_ptr += packed_stride;
        memcpy(packed_ptr, src_ptr1, non_zero_size);
        memset(packed_ptr + non_zero_rows, 0, zero_size);
        packed_ptr += packed_stride;
        memcpy(packed_ptr, src_ptr2, non_zero_size);
        memset(packed_ptr + non_zero_rows, 0, zero_size);
        packed_ptr += packed_stride;
        memcpy(packed_ptr, src_ptr3, non_zero_size);
        memset(packed_ptr + non_zero_rows, 0, zero_size);
      }
    }
  }
};

// Derived from the function of the same name in Ruy:
// https://github.com/google/ruy/blob/c9f5f9cecde3d6314df6e7d91517356bf07135eb/ruy/pack.h#L130-L140
template <Path ThePath, typename FixedKernelLayout>
void RunRuyPack(Tuning tuning, const EMat& src_matrix, PEMat* packed_matrix,
                int start_col, int end_col) {
  Mat<TBitpacked> src = UneraseType<TBitpacked>(src_matrix);
  PMat<TBitpacked> packed = UneraseType<TBitpacked>(*packed_matrix);
  if (src.layout.order == Order::kColMajor) {
    LceRuyPackImpl<ThePath, FixedKernelLayout, Order::kColMajor>::Run(
        tuning, src, &packed, start_col, end_col);
  } else {
    LceRuyPackImpl<ThePath, FixedKernelLayout, Order::kRowMajor>::Run(
        tuning, src, &packed, start_col, end_col);
  }
}

}  // namespace bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_RUY_PACK_H_
