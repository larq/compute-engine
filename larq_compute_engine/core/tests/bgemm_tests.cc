#include <gmock/gmock.h>

#include <array>

#include "larq_compute_engine/core/bgemm_functor.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;
using ce::core::bitpacking_bitwidth;
using ce::core::Layout;
using ce::core::TBitpacked;

TEST(BGemmTests, BinaryInnerProd) {
  const auto a = static_cast<TBitpacked>(0b01101110000111111101011001101000);
  const auto b = static_cast<TBitpacked>(0b01100110000110111001011011101001);
  // a and b are off by five bits so POP_CNT(a XOR b) = 5
  const auto expected = static_cast<std::int32_t>(bitpacking_bitwidth - 2 * 5);
  auto c = ce::core::compute_binary_inner_prod(a, b);
  EXPECT_EQ(c, expected);
}

template <class TBgemmFunctor, int m, int n, int k>
void test_bgemm() {
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::array<TBitpacked, a_size> a;
  a.fill(1);

  std::array<TBitpacked, b_size> b;
  b.fill(1);

  // each row of matrix "a" and column of "b" contains k same values so
  // a[i, k] XOR b[k, j] = 0 and therefore
  // c[i, j] = k * (bitpacking_bitwidth - 2 * POP_CNT(0))
  std::int32_t expected_value = k * bitpacking_bitwidth;
  std::array<std::int32_t, c_size> expected;
  expected.fill(expected_value);

  std::array<std::int32_t, c_size> c;
  TBgemmFunctor bgemm_functor;
  bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);
  EXPECT_THAT(c, ::testing::ElementsAreArray(expected));
}

TEST(BGemmTests, BGemmTestRowMajor) {
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<Layout::RowMajor, Layout::RowMajor,
                                      std::int32_t>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<BGemmFunctor, m, n, k>();
}

TEST(BGemmTests, BGemmTestColMajor) {
  using BGemmFunctor =
      ce::core::ReferenceBGemmFunctor<Layout::RowMajor, Layout::ColMajor,
                                      std::int32_t>;
  const int m = 20;
  const int k = 200;
  const int n = 30;
  test_bgemm<BGemmFunctor, m, n, k>();
}

}  // end namespace testing
}  // end namespace compute_engine
