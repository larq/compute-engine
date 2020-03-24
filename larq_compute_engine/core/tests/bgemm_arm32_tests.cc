#include <gmock/gmock.h>

#include <array>

#include "larq_compute_engine/core/arm32/bgemm.h"
#include "larq_compute_engine/core/bgemm_functor.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;
using ce::core::Layout;

TEST(BGemmTests, BGemmArm32) {
  ce::core::ReferenceBGemmFunctor<uint64_t, Layout::RowMajor, std::uint64_t,
                                  Layout::ColMajor, std::int32_t>
      bgemm_functor;

  const int m = 10;
  const int k = 20;
  const int n = 30;

  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::array<uint64_t, a_size> a;
  std::array<uint64_t, b_size> b;

  // Fill with random bits
  for (size_t i = 0; i < a_size; ++i) {
    a[i] = rand();
  }
  for (size_t i = 0; i < b_size; ++i) {
    b[i] = rand();
  }

  std::array<int32_t, c_size> c_expected;
  bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c_expected.data(), ldc);

  std::array<int32_t, c_size> c;
  ce::core::bgemm_arm32(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);

  // TODO: Enable this after we implemented bgemm_arm32
  // EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

}  // end namespace testing
}  // end namespace compute_engine
