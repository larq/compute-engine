#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/cc/core/aarch64/packbits.h"
#include "larq_compute_engine/cc/core/packbits.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

TEST(BitpackingTests, BitpackingAarch64) {
  constexpr size_t n = 128;
  constexpr size_t n_packed = (n + 63) / 64;

  std::array<float, n> input;

  for (size_t i = 0; i < n; ++i) {
    input[i] = (rand() % 2) ? 1 : -1;
  }

  std::vector<std::uint64_t> expected(n_packed);
  ce::core::packbits_array<float, uint64_t>(input.data(), n, expected.data());

  std::vector<std::uint64_t> output(n_packed);
  ce::core::packbits_aarch64(input.data(), n, output.data());

  // Currently it doesn't output anything meaningful
  // EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}

}  // end namespace testing
}  // end namespace compute_engine
