#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/bitpack.h"
#include "larq_compute_engine/core/bitpack_aarch64.h"
#include "larq_compute_engine/core/types.h"

namespace compute_engine {
namespace testing {

using namespace compute_engine::core;

void test_bitpacking(int num_4x32_blocks) {
  const int num_blocks = 4 * num_4x32_blocks;
  const int n = 32 * num_blocks;

  float input[n];
  TBitpacked output[num_blocks];
  for (auto i = 0; i < n; ++i) {
    // Try to get the position of bit i by packing the one-hot vector e_i
    for (auto j = 0; j < n; ++j) {
      if (j == i)
        input[j] = -1.2345f;
      else
        input[j] = 1.2345f;
    }
    // Run bitpacking
    bitpack_aarch64_4x32(input, num_blocks, output);
    // See where in the output the bit has popped up
    int bit_index = -1;
    int bits_found = 0;
    for (auto j = 0; j < num_blocks; ++j) {
      for (auto k = 0; k < 32; ++k) {
        if (output[j] & (1uL << k)) {
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

TEST(BitpackingAarch64, 1x4x32) { test_bitpacking(1); }
TEST(BitpackingAarch64, 2x4x32) { test_bitpacking(2); }
TEST(BitpackingAarch64, 3x4x32) { test_bitpacking(3); }
TEST(BitpackingAarch64, 11x4x32) { test_bitpacking(11); }
TEST(BitpackingAarch64, 17x4x32) { test_bitpacking(17); }

}  // end namespace testing
}  // end namespace compute_engine
