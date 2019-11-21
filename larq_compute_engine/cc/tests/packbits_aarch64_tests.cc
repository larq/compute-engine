#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/cc/core/aarch64/packbits.h"
#include "larq_compute_engine/cc/core/packbits.h"

namespace compute_engine {
namespace testing {

using namespace compute_engine::core;
using namespace compute_engine::benchmarking;

template <void (*packing_func)(const float*, size_t, void*), size_t blocksize>
void test_bitpacking() {
  constexpr size_t n = 512;

  // We will simply check if every input gets mapped uniquely to one bit
  float input[n];
  uint64_t output[n / 64];
  int mapping[n];
  for (auto i = 0; i < n; ++i) {
    mapping[i] = -1;
  }
  for (auto i = 0; i < n; ++i) {
    // Try to get the position of bit i by packing the one-hot vector e_i
    for (auto j = 0; j < n; ++j) {
      if (j == i)
        input[j] = 1.2345f;
      else
        input[j] = -1.2345f;
    }
    // Run bitpacking
    packing_func(input, n / blocksize, output);
    // See where in the output the bit has popped up
    int bit_index = -1;
    int bits_found = 0;
    for (auto j = 0; j < n / 64; ++j) {
      for (auto k = 0; k < 64; ++k) {
        if (output[j] & (1uL << k)) {
          bit_index = k + j * 64;
          bits_found++;
        }
      }
    }
    // We should have exactly one enabled bit
    EXPECT_EQ(bits_found, 1);
    // This location should *not* have been used already
    EXPECT_EQ(mapping[bit_index], -1);
  }
}

TEST(BitpackingAarch64, 64Blocks) {
  test_bitpacking<packbits_aarch64_64, 64>();
}

TEST(BitpackingAarch64, 64BlocksV2) {
  test_bitpacking<packbits_aarch64_64_v2, 64>();
}

TEST(BitpackingAarch64, 64BlocksV3) {
  test_bitpacking<packbits_aarch64_64_v3, 64>();
}

TEST(BitpackingAarch64, 128Blocks) {
  test_bitpacking<packbits_aarch64_128, 128>();
}

}  // end namespace testing
}  // end namespace compute_engine
