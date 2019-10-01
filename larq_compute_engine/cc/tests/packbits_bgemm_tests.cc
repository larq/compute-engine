#include <gmock/gmock.h>

#include <array>
#include <vector>

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "larq_compute_engine/cc/core/packbits.h"
#include "larq_compute_engine/cc/utils/macros.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

TEST(BitpackingBGEMMTests, WithBitPadding) {
  using TBitpacked = uint8_t;
  using T = float;
  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;

  const int a_num_rows = 2;
  const int a_num_cols = 9;
  std::vector<T> a_data_float{1, 1,  -1, 1,  -1, -1, -1, 1, 1,
                              1, -1, 1,  -1, -1, -1, 1,  1, -1};
  const int b_num_rows = 9;
  const int b_num_cols = 2;
  std::vector<T> b_data_float{-1, 1, 1, -1, -1, -1, 1, -1, 1,
                              -1, 1, 1, -1, 1,  1,  1, 1,  -1};

  //  3 = -1 + 1 + 1 + 1 - 1 - 1 + 1 + 1 + 1
  // -1 =  1 - 1 + 1 - 1 + 1 - 1 - 1 + 1 - 1
  // -7 = -1 - 1 - 1 - 1 - 1 - 1 - 1 + 1 - 1
  //  5 =  1 + 1 - 1 + 1 + 1 - 1 + 1 + 1 + 1
  std::vector<T> c_expected{3, -1, -7, 5};

  // 0b10001011, 0b00000001
  // 0b11000101, 0b00000000
  std::vector<TBitpacked> a_data_bpacked;
  size_t a_num_rows_bp = 0, a_num_cols_bp = 0;
  size_t a_bitpadding = 0;
  ce::core::packbits_matrix(a_data_float, a_num_rows, a_num_cols,
                            a_data_bpacked, a_num_rows_bp, a_num_cols_bp,
                            a_bitpadding,
                            ce::core::Axis::RowWise);

  // 0b10111010, 0b11100001
  // 0b00000001, 0b00000000
  std::vector<TBitpacked> b_data_bpacked;
  size_t b_num_rows_bp = 0, b_num_cols_bp = 0;
  size_t b_bitpadding = 0;
  ce::core::packbits_matrix(b_data_float, b_num_rows, b_num_cols,
                            b_data_bpacked, b_num_rows_bp, b_num_cols_bp,
                            b_bitpadding,
                            ce::core::Axis::ColWise);

  EXPECT_EQ(a_bitpadding, b_bitpadding);

  const int m = a_num_rows_bp;
  const int k = a_num_cols_bp;
  const int n = b_num_cols_bp;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int c_size = m * n;
  std::vector<T> c(c_size);
  TBGemmFunctor bgemm_functor;
  bgemm_functor(m, n, k, a_data_bpacked.data(), lda, b_data_bpacked.data(), ldb,
                c.data(), ldc, a_bitpadding);

  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

}  // end namespace testing
}  // end namespace compute_engine
