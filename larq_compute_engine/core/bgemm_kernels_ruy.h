#ifndef COMPUTE_ENGINE_CORE_BGEMM_KERNELS_RUY_H_
#define COMPUTE_ENGINE_CORE_BGEMM_KERNELS_RUY_H_

#include "larq_compute_engine/core/bgemm_functor.h"
#include "larq_compute_engine/core/bitpack_utils.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/ruy.h"

namespace compute_engine {
namespace tflite {

namespace ce = compute_engine;

using ce::core::bitpacking_bitwidth;
using ce::core::TBitpacked;

template <ruy::Path ThePath, typename DstScalar, typename Spec>
struct BgemmKernel {};

// TODO: this is hacky
#if RUY_PLATFORM_NEON
#include "bgemm_kernels_arm.h"
#endif

template <typename DstScalar, typename Spec>
struct BgemmKernel<ruy::Path::kStandardCpp, DstScalar, Spec> {
  using AccumScalar = typename Spec::AccumScalar;
  using LhsLayout = typename Spec::StandardCppKernelLhsLayout;
  using RhsLayout = typename Spec::StandardCppKernelRhsLayout;
  explicit BgemmKernel(ruy::Tuning) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const Spec& spec, int start_row, int start_col, int end_row,
           int end_col, ruy::Mat<DstScalar>* dst) const {
    const OutputTransform<DstScalar>& output_transform = spec.output_transform;

    int clamped_end_row = std::min(end_row, dst->layout.rows);
    int clamped_end_col = std::min(end_col, dst->layout.cols);
    RUY_DCHECK_LE(0, start_row);
    RUY_DCHECK_LE(start_row, clamped_end_row);
    RUY_DCHECK_LE(clamped_end_row, dst->layout.rows);
    RUY_DCHECK_LE(clamped_end_row, end_row);
    RUY_DCHECK_LE(end_row - clamped_end_row, LhsLayout::kCols);
    RUY_DCHECK_LE(0, start_col);
    RUY_DCHECK_LE(start_col, clamped_end_col);
    RUY_DCHECK_LE(clamped_end_col, dst->layout.cols);
    RUY_DCHECK_LE(clamped_end_col, end_col);
    RUY_DCHECK_LE(end_col - clamped_end_col, RhsLayout::kCols);

    ruy::profiler::ScopeLabel label("Binary Kernel (Standard Cpp)");
    const int depth = lhs.layout.rows;
    for (int i = start_row; i < clamped_end_row; i++) {
      for (int j = start_col; j < clamped_end_col; j++) {
        using AccumScalar = typename Spec::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          TBitpacked lhs_val = Element(lhs, k, i);
          TBitpacked rhs_val = Element(rhs, k, j);
          accum += ce::core::xor_popcount(lhs_val, rhs_val);
        }
        // Post-process the accumulated value and store the result
        *ElementPtr(dst, i, j) = output_transform.Run(accum, i);
      }
    }
  }
};

// A template specialisation for writing bitpacked output.
template <typename Spec>
struct BgemmKernel<ruy::Path::kStandardCpp, TBitpacked, Spec> {
  using AccumScalar = typename Spec::AccumScalar;
  using LhsLayout = typename Spec::StandardCppKernelLhsLayout;
  using RhsLayout = typename Spec::StandardCppKernelRhsLayout;
  explicit BgemmKernel(ruy::Tuning) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const Spec& spec, int start_row, int start_col, int end_row,
           int end_col, ruy::Mat<TBitpacked>* dst) const {
    const OutputTransform<TBitpacked>& output_transform = spec.output_transform;

    // We are writing bitpacked output (where we bitpack along the channel axis)
    // and so we need to operate on blocks of `bitwidth` channels at a time. As
    // the destination is column major, this means blocks of `bitwidth` rows at
    // a time. Thus, we require the LHS layout columns to be a multiple of
    // `bitwidth`.
    static_assert(LhsLayout::kCols % bitpacking_bitwidth == 0,
                  "When writing bitpacked output, the LHS layout must have a "
                  "multiple of `bitwidth` columns.");

    // Note that the rows in all these calculations are the *unpacked* rows.

    int clamped_end_row = std::min(end_row, dst->layout.rows);
    int clamped_end_col = std::min(end_col, dst->layout.cols);
    RUY_DCHECK_LE(0, start_row);
    RUY_DCHECK_LE(start_row, clamped_end_row);
    RUY_DCHECK_LE(clamped_end_row, dst->layout.rows);
    RUY_DCHECK_LE(clamped_end_row, end_row);
    RUY_DCHECK_LE(0, start_col);
    RUY_DCHECK_LE(start_col, clamped_end_col);
    RUY_DCHECK_LE(clamped_end_col, dst->layout.cols);
    RUY_DCHECK_LE(clamped_end_col, end_col);
    RUY_DCHECK_LE(end_col - clamped_end_col, RhsLayout::kCols);

    RUY_DCHECK_EQ(dst->layout.order, Order::kColMajor);

    ruy::profiler::ScopeLabel label(
        "Binary Kernel (Standard Cpp) Bitpacked Output.");

    const int depth = lhs.layout.rows;
    const int dst_stride_bitpacked =
        ce::core::GetBitpackedSize(dst->layout.stride);

    // The destination is column major and we need to bitpack along the row
    // (channels) axis so we need to loop over column index then row index.
    for (int j = start_col; j < clamped_end_col; j++) {
      TBitpacked bitpacked_column = 0;
      for (int i = start_row; i < clamped_end_row; i++) {
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          TBitpacked lhs_val = Element(lhs, k, i);
          TBitpacked rhs_val = Element(rhs, k, j);
          accum += ce::core::xor_popcount(lhs_val, rhs_val);
        }
        bool bit = output_transform.Run(accum, i);
        if (bit) {
          bitpacked_column |= TBitpacked(1)
                              << ((i - start_row) % bitpacking_bitwidth);
        }
        if (((i - start_row + 1) % bitpacking_bitwidth == 0) ||
            (i + 1 == clamped_end_row)) {
          *(dst->data.get() + i / bitpacking_bitwidth +
            j * dst_stride_bitpacked) = bitpacked_column;
          bitpacked_column = 0;
        }
      }
    }
  }
};

template <ruy::Path ThePath, typename DstScalar, typename Spec>
void RunBgemmKernelTyped(ruy::Tuning tuning, const ruy::PMat<TBitpacked>& lhs,
                         const ruy::PMat<TBitpacked>& rhs, const Spec& spec,
                         int start_row, int start_col, int end_row, int end_col,
                         ruy::Mat<DstScalar>* dst) {
  using BKernel = BgemmKernel<ThePath, DstScalar, Spec>;
  BKernel kernel(tuning);
  using LhsLayout = typename BKernel::LhsLayout;
  using RhsLayout = typename BKernel::RhsLayout;
  // end_row and end_col may be larger than dst dimensions.
  // that is because kernels write directly to the destination matrix, whose
  // dimensions may not be a multiple of the kernel dimensions, and we try to
  // keep this annoyance localized as an implementation detail in kernels,
  // by allowing to pass rounded-up values down as far as possible.
  // These assertions encode the contract.
  RUY_DCHECK_LE(0, start_row);
  RUY_DCHECK_LE(start_row, end_row);
  RUY_DCHECK_LT(end_row, dst->layout.rows + LhsLayout::kCols);
  RUY_DCHECK_EQ((end_row - start_row) % LhsLayout::kCols, 0);
  RUY_DCHECK_LE(0, start_col);
  RUY_DCHECK_LE(start_col, end_col);
  RUY_DCHECK_LT(end_col, dst->layout.cols + RhsLayout::kCols);
  RUY_DCHECK_EQ((end_col - start_col) % RhsLayout::kCols, 0);
#if RUY_OPT(FAT_KERNEL)
  kernel.Run(lhs, rhs, spec, start_row, start_col, end_row, end_col, dst);
#else
  for (int col = start_col; col < end_col; col += RhsLayout::kCols) {
    int block_end_col = std::min(col + RhsLayout::kCols, end_col);
    for (int row = start_row; row < end_row; row += LhsLayout::kCols) {
      int block_end_row = std::min(row + LhsLayout::kCols, end_row);
      kernel.Run(lhs, rhs, spec, row, col, block_end_row, block_end_col, dst);
    }
  }
#endif
}

template <ruy::Path ThePath, typename DstScalar, typename Spec>
void RunBgemmKernel(ruy::Tuning tuning, const ruy::SidePair<ruy::PEMat>& src,
                    void* spec, const ruy::SidePair<int>& start,
                    const ruy::SidePair<int>& end, ruy::EMat* dst) {
  ruy::Mat<DstScalar> mdst = ruy::UneraseType<DstScalar>(*dst);
  RunBgemmKernelTyped<ThePath, DstScalar, Spec>(
      tuning, ruy::UneraseType<TBitpacked>(src[ruy::Side::kLhs]),
      ruy::UneraseType<TBitpacked>(src[ruy::Side::kRhs]),
      *static_cast<const Spec*>(spec), start[ruy::Side::kLhs],
      start[ruy::Side::kRhs], end[ruy::Side::kLhs], end[ruy::Side::kRhs],
      &mdst);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_KERNELS_RUY_H_
