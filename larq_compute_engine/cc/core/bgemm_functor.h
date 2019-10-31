#ifndef COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_
#define COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_

#include <bitset>
#include <cstdint>
#include <limits>

#include "larq_compute_engine/cc/utils/macros.h"

namespace compute_engine {
namespace core {

enum class Layout { RowMajor, ColMajor };

template <class TIn, class TOut>
inline auto compute_binary_inner_prod(const TIn& a, const TIn& b) -> TOut {
  static_assert(std::is_unsigned<TIn>::value,
                "Input of binary inner product should be of an unsigned type.");
  static_assert(std::is_signed<TOut>::value,
                "Output of binary inner product should be of a signed type.");

  constexpr auto bitwidth = std::numeric_limits<TIn>::digits;
  std::bitset<bitwidth> bs(a ^ b);
  return bitwidth - 2 * static_cast<TOut>(bs.count());
}

template <>
inline std::int32_t compute_binary_inner_prod<std::uint8_t, std::int32_t>(
    const std::uint8_t& a, const std::uint8_t& b) {
  // TODO: __builtin_popcount works only with GCC compiler -> implement a
  // generalized version.
  // TODO: SIMD optimization for popcount
  constexpr auto bitwidth = std::numeric_limits<std::uint8_t>::digits;
  return bitwidth - 2 * static_cast<std::int32_t>(__builtin_popcount(a ^ b));
}

template <>
inline std::int32_t compute_binary_inner_prod<std::uint32_t, std::int32_t>(
    const std::uint32_t& a, const std::uint32_t& b) {
  // TODO: __builtin_popcount works only with GCC compiler -> implement a
  // generalized version.
  // TODO: SIMD optimization for popcount
  constexpr auto bitwidth = std::numeric_limits<std::uint32_t>::digits;
  return bitwidth - 2 * static_cast<std::int32_t>(__builtin_popcountl(a ^ b));
}

template <>
inline std::int32_t compute_binary_inner_prod<std::uint64_t, std::int32_t>(
    const std::uint64_t& a, const std::uint64_t& b) {
  // TODO: __builtin_popcount works only with GCC compiler -> implement a
  // generalized version.
  // TODO: SIMD optimization for popcount
  constexpr auto bitwidth = std::numeric_limits<std::uint64_t>::digits;
  return bitwidth - 2 * static_cast<std::int32_t>(__builtin_popcountll(a ^ b));
}

// A naive implementation of binary matrix multiplication, useful for
// debugging and understanding the algorithm.
template <class TIn1, Layout SIn1, class TIn2, Layout SIn2, class TOut>
class ReferenceBGemmFunctor {
 public:
  void operator()(const size_t m, const size_t n, const size_t k, const TIn1* a,
                  const size_t lda, const TIn2* b, const size_t ldb, TOut* c,
                  const size_t ldc, const int bitpaddding = 0) {
    static_assert(SIn1 == Layout::RowMajor,
                  "Left-hand input should be stored row major.");

    // for now accept only unsigned {8,32,64}-bits values as input.
    static_assert(std::is_same<TIn2, TIn1>::value,
                  "Inputs to BGEMM should have the same type.");
    static_assert(
        std::is_unsigned<TIn1>::value && std::is_integral<TIn1>::value,
        "Input to BGEMM should be of type unsigned integral.");
    static_assert(std::is_signed<TOut>::value,
                  "Output of BGEMM should be of a signed type.");

    const size_t a_i_stride = lda;
    const size_t a_l_stride = 1;
    const size_t b_j_stride = (SIn2 == Layout::RowMajor ? 1 : ldb);
    const size_t b_l_stride = (SIn2 == Layout::RowMajor ? ldb : 1);
    const size_t c_i_stride = ldc;
    const size_t c_j_stride = 1;

    size_t i, j, l;
    // The j-loop should be the inner loop for weight-stationary computations
    for (i = 0; i < m; ++i) {
      for (j = 0; j < n; ++j) {
        if (SIn2 == Layout::RowMajor) {
          int32_t total = 0;
          for (l = 0; l < k; ++l) {
            const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
            const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
            total +=
                compute_binary_inner_prod<TIn1, int32_t>(a[a_index], b[b_index]);
          }
          const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
          c[c_index] = static_cast<TOut>(total - bitpaddding);
        } else {  // ColMajor
          int32_t total = 0;
          // Because of this nice memory layout we can just increment pointers
          const TIn1* a_ptr = &a[i * a_i_stride];
          const TIn1* b_ptr = &b[j * b_j_stride];
          for (l = 0; l < k; ++l) {
            total +=
                compute_binary_inner_prod<TIn1, int32_t>(*a_ptr++, *b_ptr++);
          }
          const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
          c[c_index] = static_cast<TOut>(total - bitpaddding);
        }
      }
    }
  }
};

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_KERNELS_BGEMM_FUNCTORS_H_
