#include <gmock/gmock.h>

#include <array>
#include <vector>

#include "larq_compute_engine/cc/core/fused_bgemm_functor.h"
#include "larq_compute_engine/cc/utils/macros.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

TEST(BitpackingBGEMMTests, WithBitPadding) {
  using TBitpacked = uint8_t;
  using T = float;

  const int a_num_rows = 2;
  const int a_num_cols = 9;
  std::vector<T> a_data_float{1, 1,  -1, 1,  -1, -1, -1, 1, 1,
                              1, -1, 1,  -1, -1, -1, 1,  1, -1};
  // const int b_num_rows = 9;
  const int b_num_cols = 2;
  std::vector<T> b_data_float{-1, 1, 1, -1, -1, -1, 1, -1, 1,
                              -1, 1, 1, -1, 1,  1,  1, 1,  -1};

  //  3 = -1 + 1 + 1 + 1 - 1 - 1 + 1 + 1 + 1
  // -1 =  1 - 1 + 1 - 1 + 1 - 1 - 1 + 1 - 1
  // -7 = -1 - 1 - 1 - 1 - 1 - 1 - 1 + 1 - 1
  //  5 =  1 + 1 - 1 + 1 + 1 - 1 + 1 + 1 + 1
  std::vector<T> c_expected{3, -1, -7, 5};

  const int m = a_num_rows;
  const int k = a_num_cols;
  const int n = b_num_cols;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, T, T, TBitpacked, TBGemmFunctor>;

  const int c_size = m * n;
  std::vector<T> c(c_size);
  TFusedBGemmFunctor fused_bgemm_functor;
  fused_bgemm_functor(m, n, k, a_data_float.data(), lda, b_data_float.data(),
                      ldb, c.data(), ldc);

  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

}  // end namespace testing
}  // end namespace compute_engine
