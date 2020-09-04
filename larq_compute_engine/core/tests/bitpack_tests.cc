#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "larq_compute_engine/core/bitpack.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

using ce::core::bitpacking_bitwidth;
using ce::core::TBitpacked;

class BitpackingTest
    : public ::testing::TestWithParam<std::tuple<int, int, std::int32_t>> {};

template <typename TIn>
void runBitpackingTest(const int num_rows, const int num_cols,
                       const std::int32_t zero_point) {
  if (std::is_same<TIn, float>::value && zero_point != 0) {
    GTEST_SKIP();
  }

  const int num_packed_cols = ce::core::GetBitpackedSize(num_cols);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<TIn> input_matrix(num_rows * num_cols);
  std::vector<TBitpacked> output_matrix(
      ce::core::GetBitpackedMatrixSize(num_rows, num_cols));

  // Generate some random data for the input.
  if (std::is_same<TIn, float>::value) {
    std::generate(std::begin(input_matrix), std::end(input_matrix), [&gen]() {
      return std::uniform_real_distribution<>(-1.5, 1.5)(gen);
    });
  } else if (std::is_same<TIn, std::int8_t>::value) {
    std::generate(std::begin(input_matrix), std::end(input_matrix), [&gen]() {
      return std::uniform_real_distribution<>(-128, 127)(gen);
    });
  } else {
    EXPECT_FALSE(true);
  }

  // Perform the bitpacking.
  ce::core::bitpack_matrix(input_matrix.data(), num_rows, num_cols,
                           output_matrix.data(), zero_point);

  // Verify correctness of the results.
  for (auto i = 0; i < num_rows; i++) {
    for (auto j = 0; j < bitpacking_bitwidth * num_packed_cols; j++) {
      const bool packed_bit =
          output_matrix.at(i * num_packed_cols + j / bitpacking_bitwidth) &
          (TBitpacked(1) << (j % bitpacking_bitwidth));

      if (j < num_cols) {
        // If this bit position corresponds to an actual value, compare against
        // the sign of that value...
        bool expected_bit;
        if (std::is_same<TIn, float>::value) {
          assert(zero_point == 0);
          expected_bit = input_matrix.at(i * num_cols + j) < 0;
        } else {
          expected_bit = static_cast<std::int32_t>(
                             input_matrix.at(i * num_cols + j)) < zero_point;
        }
        EXPECT_EQ(packed_bit, expected_bit);
      } else {
        // ...otherwise it's a 'padding' bit, and we expect it to be zero.
        EXPECT_EQ(packed_bit, 0);
      }
    }
  }
}

TEST_P(BitpackingTest, BitpackFloats) {
  runBitpackingTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                           std::get<2>(GetParam()));
}

TEST_P(BitpackingTest, BitpackInt8) {
  runBitpackingTest<std::int8_t>(std::get<0>(GetParam()),
                                 std::get<1>(GetParam()),
                                 std::get<2>(GetParam()));
}

std::string TestName(
    const ::testing::TestParamInfo<BitpackingTest::ParamType>& info) {
  // We have to treat the zero point specially, because we can't have a
  // hyphen in the name, and so can't naturally represent negative numbers.
  std::string param_zp_value_str =
      absl::StrCat(std::get<2>(info.param) >= 0 ? "Pos" : "Neg",
                   std::abs(std::get<2>(info.param)));
  return absl::StrCat("Rows_", std::get<0>(info.param), "__Cols_",
                      std::get<1>(info.param), "__ZeroPoint_",
                      param_zp_value_str);
}

INSTANTIATE_TEST_SUITE_P(Bitpacking, BitpackingTest,
                         ::testing::Combine(
                             // num_rows
                             ::testing::Values(1, 2, 3, 8, 10, 15, 64),
                             // num_cols
                             ::testing::Values(1, 3, 16, 32, 33, 63, 64, 128),
                             // zero_point
                             ::testing::Values(-1000, -1, 0, 23, 127, 128)),
                         TestName);

}  // end namespace testing
}  // namespace compute_engine
