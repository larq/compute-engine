#include <gmock/gmock.h>

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "larq_compute_engine/core/fused_bgemm_functor.h"
#include "larq_compute_engine/core/macros.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;
using ce::core::Layout;

template <typename TBitpacked, typename T>
void fused_test_rowmajor() {
  // clang-format off
  const int a_num_rows = 2;
  const int a_num_cols = 9;
  std::vector<T> a_data_float{1,  1, -1,  1, -1, -1, -1,  1,  1,
                              1, -1,  1, -1, -1, -1,  1,  1, -1};
  //const int b_num_rows = 9;
  const int b_num_cols = 2;
  std::vector<T> b_data_float{ -1,  1,
                                1, -1,
                               -1, -1,
                                1, -1,
                                1, -1,
                                1,  1,
                               -1,  1,
                                1,  1,
                                1, -1};
  // clang-format on

  // Both matrices above are stored row major so they
  // are as they are shaped just as they appear in the code.
  //
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
      ce::core::ReferenceBGemmFunctor<TBitpacked, Layout::RowMajor, TBitpacked,
                                      Layout::RowMajor, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, Layout::RowMajor, T, Layout::RowMajor, T,
                                  TBitpacked, TBGemmFunctor>;

  const int c_size = m * n;
  std::vector<T> c(c_size);
  TFusedBGemmFunctor fused_bgemm_functor;
  fused_bgemm_functor(m, n, k, a_data_float.data(), lda, b_data_float.data(),
                      ldb, c.data(), ldc);

  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

template <typename TBitpacked, typename T>
void fused_test_colmajor() {
  // clang-format off
  const int a_num_rows = 2;
  const int a_num_cols = 9;
  std::vector<T> a_data_float{1,  1, -1,  1, -1, -1, -1,  1,  1,
                              1, -1,  1, -1, -1, -1,  1,  1, -1};
  const int b_num_rows = 9;
  const int b_num_cols = 2;
  std::vector<T> b_data_float{-1,  1, -1,  1,  1,  1, -1, 1,  1,
                               1, -1, -1, -1, -1,  1,  1, 1, -1};
  // clang-format on

  // The `b` matrix is the transposed version of the one in the row major
  // test so the output should be the same.

  //  3 = -1 + 1 + 1 + 1 - 1 - 1 + 1 + 1 + 1
  // -1 =  1 - 1 + 1 - 1 + 1 - 1 - 1 + 1 - 1
  // -7 = -1 - 1 - 1 - 1 - 1 - 1 - 1 + 1 - 1
  //  5 =  1 + 1 - 1 + 1 + 1 - 1 + 1 + 1 + 1
  std::vector<T> c_expected{3, -1, -7, 5};

  const int m = a_num_rows;
  const int k = a_num_cols;
  const int n = b_num_cols;
  const int lda = k;
  const int ldb = b_num_rows;
  const int ldc = n;

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, Layout::RowMajor, TBitpacked,
                                      Layout::ColMajor, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, Layout::RowMajor, T, Layout::ColMajor, T,
                                  TBitpacked, TBGemmFunctor>;

  const int c_size = m * n;
  std::vector<T> c(c_size);
  TFusedBGemmFunctor fused_bgemm_functor;
  fused_bgemm_functor(m, n, k, a_data_float.data(), lda, b_data_float.data(),
                      ldb, c.data(), ldc);

  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

template <typename TBitpacked, typename T>
void fused_test_eigen(int m, int n, int k) {
  using MatRowMajor =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatColMajor =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  const int a_num_rows = m;
  const int a_num_cols = k;
  const int b_num_rows = k;
  const int b_num_cols = n;

  MatRowMajor a(a_num_rows, a_num_cols);
  MatColMajor b(b_num_rows, b_num_cols);

  // Fill with random values
  for (int i = 0; i < a_num_rows; ++i) {
    for (int j = 0; j < a_num_cols; ++j) {
      a(i, j) = (rand() % 2) ? 1 : -1;
    }
  }
  for (int i = 0; i < b_num_rows; ++i) {
    for (int j = 0; j < b_num_cols; ++j) {
      b(i, j) = (rand() % 2) ? 1 : -1;
    }
  }

  MatRowMajor c_correct = a * b;
  const T* c_expected = c_correct.data();

  const int lda = k;
  const int ldb = b_num_rows;
  const int ldc = n;

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, Layout::RowMajor, TBitpacked,
                                      Layout::ColMajor, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, Layout::RowMajor, T, Layout::ColMajor, T,
                                  TBitpacked, TBGemmFunctor>;

  const int c_size = m * n;
  std::vector<T> c(c_size);
  TFusedBGemmFunctor fused_bgemm_functor;
  fused_bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);

  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected, (size_t)c_size));
}

TEST(BitpackingBGEMMTests, RowMajorWithBitPadding8) {
  fused_test_rowmajor<uint8_t, float>();
}

TEST(BitpackingBGEMMTests, RowMajorWithBitPadding64) {
  fused_test_rowmajor<uint64_t, float>();
}

TEST(BitpackingBGEMMTests, ColMajorWithBitPadding8) {
  fused_test_colmajor<uint8_t, float>();
}

TEST(BitpackingBGEMMTests, ColMajorWithBitPadding64) {
  fused_test_colmajor<uint64_t, float>();
}

TEST(BitpackingBGEMMTests, ColMajorEigen) {
  fused_test_eigen<uint64_t, float>(5, 7, 11);
}

}  // end namespace testing
}  // end namespace compute_engine
