#ifndef COMPUTE_ENGINE_CORE_TYPES_H_
#define COMPUTE_ENGINE_CORE_TYPES_H_

#include <cstdint>
#include <limits>
#include <type_traits>

namespace compute_engine {
namespace core {

// Define these once here, so they can be included everywhere.
using TBitpacked = std::int32_t;
constexpr std::size_t bitpacking_bitwidth =
    std::numeric_limits<typename std::make_unsigned<TBitpacked>::type>::digits;

// defines the memory layout of the filter values
enum class FilterFormat { Unknown, HWIO, OHWI, OHWI_PACKED };

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_TYPES_H_
