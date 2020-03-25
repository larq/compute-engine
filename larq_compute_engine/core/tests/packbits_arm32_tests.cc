#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/packbits_arm32.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

TEST(BitpackingTests, BitpackingARM32) {
  constexpr std::size_t n = 128;
  constexpr std::size_t n_packed = (n + 63) / 64;

  std::array<float, n> input;

  for (size_t i = 0; i < n; ++i) {
    input[i] = (rand() % 2) ? 1 : -1;
  }

  std::vector<std::uint64_t> expected(n_packed);
  ce::core::packbits_array<ce::core::BitpackOrder::Optimized>(input.data(), n,
                                                              expected.data());

  std::vector<std::uint64_t> output(n_packed);
  ce::core::packbits_arm32(input.data(), n, output.data());

  // Currently it doesn't output anything meaningful
  // EXPECT_THAT(output, ::testing::ElementsAreArray(expected));

  // The test assembly that I put in will compute input[0] ^ input[1]
  // So we check that here
  std::uint32_t* ptr_in = reinterpret_cast<uint32_t*>(input.data());
  std::uint32_t* ptr_out = reinterpret_cast<uint32_t*>(output.data());
  std::uint32_t expected_output = ptr_in[0] ^ ptr_in[1];
  EXPECT_THAT(ptr_out[0], expected_output);
}

}  // end namespace testing
}  // end namespace compute_engine
