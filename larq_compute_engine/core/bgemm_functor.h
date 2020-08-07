#ifndef COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_
#define COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_

#include <bitset>
#include <cstdint>
#include <limits>

#include "larq_compute_engine/core/types.h"

namespace compute_engine {
namespace core {

enum class Layout { RowMajor, ColMajor };

using compute_engine::core::bitpacking_bitwidth;
using compute_engine::core::TBitpacked;

inline std::int32_t compute_binary_inner_prod(const TBitpacked& a,
                                              const TBitpacked& b) {
  // TODO: __builtin_popcount works only with GCC compiler -> implement a
  // generalized version.
  return bitpacking_bitwidth -
         2 * static_cast<std::int32_t>(__builtin_popcount(a ^ b));
}

inline std::int32_t xor_popcount(const TBitpacked& a, const TBitpacked& b) {
  return __builtin_popcount(a ^ b);
}

// A naive implementation of binary matrix multiplication, useful for
// debugging and understanding the algorithm.
template <Layout LLhs = Layout::RowMajor, Layout LRhs = Layout::RowMajor,
          class TOut = float, Layout LOut = Layout::RowMajor,
          class TAccum = std::int32_t>
class ReferenceBGemmFunctor {
 public:
  void operator()(const std::size_t m, const std::size_t n, const std::size_t k,
                  const TBitpacked* a, const std::size_t lda,
                  const TBitpacked* b, const std::size_t ldb, TOut* c,
                  const std::size_t ldc, const int bitpaddding = 0) {
    static_assert(std::is_signed<TOut>::value,
                  "Output of BGEMM should be of a signed type.");

    const std::size_t a_i_stride = (LLhs == Layout::RowMajor ? lda : 1);
    const std::size_t a_l_stride = (LLhs == Layout::RowMajor ? 1 : lda);
    const std::size_t b_j_stride = (LRhs == Layout::RowMajor ? 1 : ldb);
    const std::size_t b_l_stride = (LRhs == Layout::RowMajor ? ldb : 1);
    const std::size_t c_i_stride = (LOut == Layout::RowMajor ? ldc : 1);
    const std::size_t c_j_stride = (LOut == Layout::RowMajor ? 1 : ldc);

    std::size_t i, j, l;
    // The j-loop should be the inner loop for weight-stationary computations
    for (i = 0; i < m; ++i) {
      for (j = 0; j < n; ++j) {
        TAccum total(0);
        for (l = 0; l < k; ++l) {
          const std::size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
          const std::size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
          total += compute_binary_inner_prod(a[a_index], b[b_index]);
        }
        const std::size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
        c[c_index] = static_cast<TOut>(total - bitpaddding);
      }  // end of j loop
    }    // end of i loop
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_
