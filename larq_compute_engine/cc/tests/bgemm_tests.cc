#include <gmock/gmock.h>

#include <array>

#include "larq_compute_engine/cc/core/bgemm_functor.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;
using ce::core::Layout;

template <class TIn, class TOut>
void test_binary_inner_prod() {
  const auto a = static_cast<TIn>(0b11101110);
  const auto b = static_cast<TIn>(0b11111101);
  // a and b are off by three bits so POP_CNT(a XOR b) = 3
  const auto expected =
      static_cast<TOut>(std::numeric_limits<TIn>::digits - 2 * 3);
  auto c = ce::core::compute_binary_inner_prod<TIn, TOut>(a, b);
  EXPECT_EQ(c, expected);
}

template <class TIn, class TOut, class TBgemmFunctor, int m, int n, int k>
void test_bgemm() {
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::array<TIn, a_size> a;
  a.fill(1);

  std::array<TIn, b_size> b;
  b.fill(1);

  // each row of matrix "a" and column of "b" contains k same values so
  // a[i, k] XOR b[k, j] = 0 and therefore
  // c[i, j] = k * (std::numeric_limits<TIn>::digits - 2 * POP_CNT(0))
  TOut expected_value = k * std::numeric_limits<TIn>::digits;
  std::array<TOut, c_size> expected;
  expected.fill(expected_value);

  std::array<TOut, c_size> c;
  TBgemmFunctor bgemm_functor;
  bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);
  EXPECT_THAT(c, ::testing::ElementsAreArray(expected));
}

TEST(BGemmTests, BinaryInnerProdUInt8) {
  using TIn = uint8_t;
  using TOut = int32_t;
  test_binary_inner_prod<TIn, TOut>();
}

TEST(BGemmTests, BinaryInnerProdUInt32) {
  using TIn = uint32_t;
  using TOut = int32_t;
  test_binary_inner_prod<TIn, TOut>();
}

TEST(BGemmTests, BinaryInnerProdUInt64) {
  using TIn = uint64_t;
  using TOut = int32_t;
  test_binary_inner_prod<TIn, TOut>();
}

TEST(BGemmTests, BGemmTestUInt8) {
  using TIn = uint8_t;
  using TOut = int32_t;
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TIn, Layout::RowMajor, TIn,
                                      Layout::RowMajor, TOut>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<TIn, TOut, BGemmFunctor, m, n, k>();
}

TEST(BGemmTests, BGemmTestUInt32) {
  using TIn = uint32_t;
  using TOut = int32_t;
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TIn, Layout::RowMajor, TIn,
                                      Layout::RowMajor, TOut>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<TIn, TOut, BGemmFunctor, m, n, k>();
}

TEST(BGemmTests, BGemmTestUInt64) {
  using TIn = uint64_t;
  using TOut = int32_t;
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TIn, Layout::RowMajor, TIn,
                                      Layout::RowMajor, TOut>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<TIn, TOut, BGemmFunctor, m, n, k>();
}

TEST(BGemmTests, BGemmTestUInt64ColMajor) {
  using TIn = uint64_t;
  using TOut = int32_t;
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TIn, Layout::RowMajor, TIn,
                                      Layout::ColMajor, TOut>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<TIn, TOut, BGemmFunctor, m, n, k>();
}

}  // end namespace testing
}  // end namespace compute_engine
