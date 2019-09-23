#include <gmock/gmock.h>

#include <array>
#include <vector>

#include "larq_compute_engine/cc/core/packbits.h"
#include "larq_compute_engine/cc/utils/macros.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

// tests with non-uniform data verify the correctness of the bitpacking
// operation itself while the uniform tests are used to verify the bitpacking
// with different bitwidths
template <class TIn>
void test_bitpacking_nonuniform_input_rowwise() {
  const auto bitpacking_axis = ce::core::Axis::RowWise;

  // input matrix (row-major memory laytout)
  const size_t num_rows = 2, num_cols = 8;
  std::array<TIn, 16> input{1, 1,  -1, 1,  -1, -1, -1, 1,
                            1, -1, 1,  -1, -1, -1, 1,  1};

  // expected output matrix after bitpacking
  const size_t expected_num_rows = 2, expected_num_cols = 1;
  std::vector<uint8_t> expected;
  if (CE_IS_BIG_ENDIAN)
    expected = {0b11010001, 0b10100011};
  else
    expected = {0b10001011, 0b11000101};

  std::vector<uint8_t> output;
  size_t num_rows_bp = 0, num_cols_bp = 0;
  ce::core::packbits_matrix(input, num_rows, num_cols, output, num_rows_bp,
                            num_cols_bp, bitpacking_axis);

  EXPECT_EQ(num_rows_bp, expected_num_rows);
  EXPECT_EQ(num_cols_bp, expected_num_cols);
  EXPECT_EQ(output.size(), num_rows_bp * num_cols_bp);
  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

template <class TIn>
void test_bitpacking_nonuniform_input_colwise() {
  const auto bitpacking_axis = ce::core::Axis::ColWise;

  // input matrix (row-major memory laytout)
  const size_t num_rows = 8, num_cols = 2;
  std::array<TIn, 16> input{1,  1,  1,  -1, -1, 1, 1, -1,
                            -1, -1, -1, -1, -1, 1, 1, 1};

  // expected output matrix after bitpacking
  const size_t expected_num_rows = 1, expected_num_cols = 2;
  std::vector<uint8_t> expected;
  if (CE_IS_BIG_ENDIAN)
    expected = {0b11010001, 0b10100011};
  else
    expected = {0b10001011, 0b11000101};

  std::vector<uint8_t> output;
  size_t num_rows_bp = 0, num_cols_bp = 0;
  ce::core::packbits_matrix(input, num_rows, num_cols, output, num_rows_bp,
                            num_cols_bp, bitpacking_axis);

  EXPECT_EQ(num_rows_bp, expected_num_rows);
  EXPECT_EQ(num_cols_bp, expected_num_cols);
  EXPECT_EQ(output.size(), num_rows_bp * num_cols_bp);
  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

template <class TIn, class TOut, size_t num_rows, size_t num_cols>
void test_bitpacking(const ce::core::Axis bitpacking_axis,
                     const size_t expected_num_rows,
                     const size_t expected_num_cols) {
  const size_t bitwidth = std::numeric_limits<TOut>::digits;

  const size_t num_elems = num_rows * num_cols;
  std::array<TIn, num_elems> input;
  input.fill(1);

  std::vector<TOut> output;
  size_t num_rows_bp = 0, num_cols_bp = 0;
  ce::core::packbits_matrix(input, num_rows, num_cols, output, num_rows_bp,
                            num_cols_bp, bitpacking_axis);

  TOut expected_value = std::numeric_limits<TOut>::max();
  const size_t num_elems_bp = num_elems / bitwidth;
  std::array<TOut, num_elems_bp> expected;
  expected.fill(expected_value);

  EXPECT_EQ(num_rows_bp, expected_num_rows);
  EXPECT_EQ(num_cols_bp, expected_num_cols);
  EXPECT_EQ(output.size(), num_rows_bp * num_cols_bp);
  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

TEST(BitpackingTests, BitpackingRowMajorUInt8NonUniformInput) {
  test_bitpacking_nonuniform_input_rowwise<float>();
}

TEST(BitpackingTests, BitpackingColMajorUInt8NonUniformInput) {
  test_bitpacking_nonuniform_input_colwise<float>();
}

TEST(BitpackingTests, BitpackingRowMajorUInt8) {
  test_bitpacking<float, uint8_t, 2, 128>(ce::core::Axis::RowWise, 2, 16);
}

TEST(BitpackingTests, BitpackingRowMajorUInt32) {
  test_bitpacking<float, uint32_t, 2, 128>(ce::core::Axis::RowWise, 2, 4);
}

TEST(BitpackingTests, BitpackingRowMajorUInt64) {
  test_bitpacking<float, uint64_t, 2, 128>(ce::core::Axis::RowWise, 2, 2);
}

TEST(BitpackingTests, BitpackingColumnMajorUInt8) {
  test_bitpacking<float, uint8_t, 128, 2>(ce::core::Axis::ColWise, 16, 2);
}

TEST(BitpackingTests, BitpackingColumnMajorUInt32) {
  test_bitpacking<float, uint32_t, 128, 2>(ce::core::Axis::ColWise, 4, 2);
}

TEST(BitpackingTests, BitpackingColumnMajorUInt64) {
  test_bitpacking<float, uint64_t, 128, 2>(ce::core::Axis::ColWise, 2, 2);
}

}  // end namespace testing
}  // end namespace compute_engine
