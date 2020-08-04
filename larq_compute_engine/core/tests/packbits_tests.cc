#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/packbits.h"

// From @flatbuffers, used for the FLATBUFFERS_LITTLEENDIAN macro
#include "flatbuffers/base.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

// tests with non-uniform data verify the correctness of the bitpacking
// operation itself while the uniform tests are used to verify the bitpacking
// with different bitwidths
template <class TIn>
void test_bitpacking_nonuniform_input() {
  // input matrix (row-major memory laytout)
  const std::size_t num_rows = 2, num_cols = 8;
  std::array<TIn, 16> input{1, 1,  -1, 1,  -1, -1, -1, 1,
                            1, -1, 1,  -1, -1, -1, 1,  1};

  // expected output matrix after bitpacking
  const std::size_t expected_num_rows = 2, expected_num_cols = 1;
  std::vector<std::uint8_t> expected;
  if (!FLATBUFFERS_LITTLEENDIAN)
    expected = {0b00101110, 0b01011100};
  else
    expected = {0b01110100, 0b00111010};

  std::vector<std::uint8_t> output(
      ce::core::GetPackedMatrixSize<std::uint8_t>(num_rows, num_cols));
  ce::core::packbits_matrix(input.data(), num_rows, num_cols, output.data());

  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

template <class TIn, class TOut, std::size_t num_rows, std::size_t num_cols>
void test_bitpacking(const std::size_t expected_num_rows,
                     const std::size_t expected_num_cols) {
  const std::size_t bitwidth = std::numeric_limits<TOut>::digits;

  const std::size_t num_elems = num_rows * num_cols;
  std::array<TIn, num_elems> input;
  input.fill(-1);

  std::vector<TOut> output(
      ce::core::GetPackedMatrixSize<TOut>(num_rows, num_cols));
  ce::core::packbits_matrix(input.data(), num_rows, num_cols, output.data());

  TOut expected_value = std::numeric_limits<TOut>::max();
  const std::size_t num_elems_bp = num_elems / bitwidth;
  std::array<TOut, num_elems_bp> expected;
  expected.fill(expected_value);

  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

TEST(BitpackingTests, BitpackingRowMajorUInt8NonUniformInput) {
  test_bitpacking_nonuniform_input<float>();
}

TEST(BitpackingTests, BitpackingRowMajorUInt8) {
  test_bitpacking<float, std::uint8_t, 2, 128>(2, 16);
}

TEST(BitpackingTests, BitpackingRowMajorUInt32) {
  test_bitpacking<float, std::uint32_t, 2, 128>(2, 4);
}

TEST(BitpackingTests, BitpackingRowMajorUInt64) {
  test_bitpacking<float, std::uint64_t, 2, 128>(2, 2);
}

TEST(BitpackingWithBitPaddingTests, RowMajorPadding) {
  // input matrix
  const int num_rows = 2;
  const int num_cols = 9;
  std::vector<float> input{-1, -1, 1,  -1, 1, 1, 1,  -1, -1,
                           -1, 1,  -1, 1,  1, 1, -1, -1, 1};

  // expected output matrix after bitpacking
  std::vector<std::uint8_t> expected;
  if (!FLATBUFFERS_LITTLEENDIAN)
    expected = {0b11010001, 0b10000000, 0b10100011, 0b00000000};
  else
    expected = {0b10001011, 0b00000001, 0b11000101, 0b00000000};

  std::vector<std::uint8_t> output(
      ce::core::GetPackedMatrixSize<std::uint8_t>(num_rows, num_cols));
  ce::core::packbits_matrix(input.data(), num_rows, num_cols, output.data());

  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
};

}  // end namespace testing
}  // end namespace compute_engine
