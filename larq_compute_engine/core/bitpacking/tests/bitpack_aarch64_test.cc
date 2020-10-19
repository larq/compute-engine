#include "larq_compute_engine/core/bitpacking/bitpack_aarch64.h"

#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/types.h"

namespace compute_engine {
namespace core {
namespace bitpacking {

template <typename DstScalar>
void test_bitpacking_order(const int num_4x32_blocks) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");

  const int num_blocks = 4 * num_4x32_blocks;
  const int n = 32 * num_blocks;

  DstScalar input[n];
  const DstScalar zero_point =
      std::is_same<DstScalar, std::int8_t>::value ? -42 : 0;
  TBitpacked output[num_blocks];
  for (auto i = 0; i < n; ++i) {
    // Try to get the position of bit i by packing the one-hot vector e_i
    for (auto j = 0; j < n; ++j) {
      if (j == i)
        input[j] = zero_point - 5;
      else
        input[j] = zero_point + 5;
    }
    // Run bitpacking
    bitpack_aarch64_4x32(input, num_blocks, output, zero_point);
    // See where in the output the bit has popped up
    int bit_index = -1;
    int bits_found = 0;
    for (auto j = 0; j < num_blocks; ++j) {
      for (auto k = 0; k < 32; ++k) {
        if (output[j] & (TBitpacked(1) << k)) {
          bit_index = k + j * 32;
          bits_found++;
        }
      }
    }

    // We should have exactly one enabled bit...
    EXPECT_EQ(bits_found, 1);
    // ...and it should be in the i^th position.
    EXPECT_EQ(bit_index, i);
  }
}

TEST(BitpackingAarch64, Float_1x4x32) { test_bitpacking_order<float>(1); }
TEST(BitpackingAarch64, Float_2x4x32) { test_bitpacking_order<float>(2); }
TEST(BitpackingAarch64, Float_3x4x32) { test_bitpacking_order<float>(3); }
TEST(BitpackingAarch64, Float_11x4x32) { test_bitpacking_order<float>(11); }
TEST(BitpackingAarch64, Float_17x4x32) { test_bitpacking_order<float>(17); }

TEST(BitpackingAarch64, Int8_1x4x32) { test_bitpacking_order<std::int8_t>(1); }
TEST(BitpackingAarch64, Int8_2x4x32) { test_bitpacking_order<std::int8_t>(2); }
TEST(BitpackingAarch64, Int8_3x4x32) { test_bitpacking_order<std::int8_t>(3); }
TEST(BitpackingAarch64, Int8_11x4x32) {
  test_bitpacking_order<std::int8_t>(11);
}
TEST(BitpackingAarch64, Int8_17x4x32) {
  test_bitpacking_order<std::int8_t>(17);
}

}  // namespace bitpacking
}  // namespace core
}  // namespace compute_engine
