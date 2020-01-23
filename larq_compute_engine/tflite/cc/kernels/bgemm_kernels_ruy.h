#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_RUY_H_

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/ruy.h"

namespace compute_engine {
namespace tflite {

namespace ce = compute_engine;

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct BgemmKernel {};

// TODO: this is hacky
#if RUY_PLATFORM(NEON)
#include "bgemm_kernels_arm.h"
#elif RUY_PLATFORM(X86)
#include "bgemm_kernels_x86.h"
#endif

template <class TIn, class TOut>
inline auto xorpopcount(const TIn& a, const TIn& b) -> TOut {
  static_assert(std::is_unsigned<TIn>::value,
                "Input of binary inner product should be of an unsigned type.");
  static_assert(std::is_signed<TOut>::value,
                "Output of binary inner product should be of a signed type.");

  constexpr auto bitwidth = std::numeric_limits<TIn>::digits;
  return std::bitset<bitwidth>(a ^ b).count();
}

template <>
inline std::int32_t xorpopcount<std::uint32_t, std::int32_t>(
    const std::uint32_t& a, const std::uint32_t& b) {
  return __builtin_popcountl(a ^ b);
}

template <>
inline std::int32_t xorpopcount<std::uint64_t, std::int32_t>(
    const std::uint64_t& a, const std::uint64_t& b) {
  return __builtin_popcountll(a ^ b);
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BgemmKernel<ruy::Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar,
                   Spec> {
  using AccumScalar = typename Spec::AccumScalar;
  using LhsLayout = typename Spec::StandardCppKernelLhsLayout;
  using RhsLayout = typename Spec::StandardCppKernelRhsLayout;
  explicit BgemmKernel(ruy::Tuning) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to binary kernel should have the same type.");
    static_assert(
        std::is_unsigned<LhsScalar>::value &&
            std::is_integral<LhsScalar>::value,
        "Input to binary kernel should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");

    using TBitpacked = LhsScalar;

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

    gemmlowp::ScopedProfilingLabel label("Binary Kernel (Standard Cpp)");
    const int depth = lhs.layout.rows;
    for (int i = start_row; i < clamped_end_row; i++) {
      for (int j = start_col; j < clamped_end_col; j++) {
        using AccumScalar = typename Spec::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          TBitpacked lhs_val = Element(lhs, k, i);
          TBitpacked rhs_val = Element(rhs, k, j);
          // accum += ce::core::compute_binary_inner_prod<TBitpacked,
          // AccumScalar>(
          //    lhs_val, rhs_val);
          accum += xorpopcount<TBitpacked, AccumScalar>(lhs_val, rhs_val);
        }
        if (spec.fused_multiply) {
          accum *= spec.fused_multiply[i];
        }
        if (spec.fused_add) {
          accum += spec.fused_add[i];
        }
        *ElementPtr(dst, i, j) = static_cast<DstScalar>(accum);
      }
    }
  }
};

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunBgemmKernelTyped(ruy::Tuning tuning,
                         const ruy::PackedMatrix<LhsScalar>& lhs,
                         const ruy::PackedMatrix<RhsScalar>& rhs,
                         const Spec& spec, int start_row, int start_col,
                         int end_row, int end_col,
                         ruy::Matrix<DstScalar>* dst) {
  using BKernel = BgemmKernel<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>;
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
#if RUY_OPT_ENABLED(RUY_OPT_FAT_KERNEL)
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

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunBgemmKernel(ruy::Tuning tuning, const ruy::SidePair<ruy::PMatrix>& src,
                    void* spec, const ruy::SidePair<int>& start,
                    const ruy::SidePair<int>& end, ruy::DMatrix* dst) {
  ruy::Matrix<DstScalar> mdst = ruy::ToMatrix<DstScalar>(*dst);
  RunBgemmKernelTyped<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>(
      tuning, ruy::ToPackedMatrix<LhsScalar>(src[ruy::Side::kLhs]),
      ruy::ToPackedMatrix<RhsScalar>(src[ruy::Side::kRhs]),
      *static_cast<const Spec*>(spec), start[ruy::Side::kLhs],
      start[ruy::Side::kRhs], end[ruy::Side::kLhs], end[ruy::Side::kRhs],
      &mdst);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_RUY_H_
