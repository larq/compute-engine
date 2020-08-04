#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/packbits_aarch64.h"

namespace compute_engine {
namespace testing {

using namespace compute_engine::core;

template <void (*packing_func)(const float*, std::size_t, std::uint64_t*),
          std::size_t block_size, std::size_t num_blocks>
void test_bitpacking() {
  constexpr std::size_t n = block_size * num_blocks;

  float input[n];
  std::uint64_t output[num_blocks];
  for (auto i = 0; i < n; ++i) {
    // Try to get the position of bit i by packing the one-hot vector e_i
    for (auto j = 0; j < n; ++j) {
      if (j == i)
        input[j] = -1.2345f;
      else
        input[j] = 1.2345f;
    }
    // Run bitpacking
    packing_func(input, num_blocks, output);
    // See where in the output the bit has popped up
    int bit_index = -1;
    int bits_found = 0;
    for (auto j = 0; j < num_blocks; ++j) {
      for (auto k = 0; k < block_size; ++k) {
        if (output[j] & (1uL << k)) {
          bit_index = k + j * block_size;
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

TEST(BitpackingAarch64, 1x64) { test_bitpacking<packbits_aarch64_64, 64, 1>(); }
TEST(BitpackingAarch64, 2x64) { test_bitpacking<packbits_aarch64_64, 64, 2>(); }
TEST(BitpackingAarch64, 3x64) { test_bitpacking<packbits_aarch64_64, 64, 3>(); }
TEST(BitpackingAarch64, 4x64) { test_bitpacking<packbits_aarch64_64, 64, 4>(); }
TEST(BitpackingAarch64, 8x64) { test_bitpacking<packbits_aarch64_64, 64, 8>(); }
TEST(BitpackingAarch64, 17x64) {
  test_bitpacking<packbits_aarch64_64, 64, 17>();
}
TEST(BitpackingAarch64, 23x64) {
  test_bitpacking<packbits_aarch64_64, 64, 23>();
}

}  // end namespace testing
}  // end namespace compute_engine
