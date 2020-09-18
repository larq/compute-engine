#ifndef COMPUTE_ENGINE_CORE_TYPES_H_
#define COMPUTE_ENGINE_CORE_TYPES_H_

#include <bitset>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace compute_engine {
namespace core {

// In our kernels we may occasionally read (but never write) beyond the end of
// an array. This is the maximum number of extra bytes that will be read, and
// should be added as padding to the end of tensor allocations.
#define LCE_EXTRA_BYTES 16

// Define these once here, so they can be included everywhere.
using TBitpacked = std::int32_t;
constexpr std::size_t bitpacking_bitwidth =
    std::numeric_limits<typename std::make_unsigned<TBitpacked>::type>::digits;

inline int xor_popcount(const TBitpacked& a, const TBitpacked& b) {
  return std::bitset<bitpacking_bitwidth>(a ^ b).count();
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_TYPES_H_
