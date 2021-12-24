#ifndef COMPUTE_ENGINE_CORE_TYPES_H_
#define COMPUTE_ENGINE_CORE_TYPES_H_

#include <bitset>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace compute_engine {
namespace core {

#if defined(__GNUC__)
#define LCE_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#define LCE_UNLIKELY(condition) (__builtin_expect(condition, 0))
#else
#define LCE_LIKELY(condition) (condition)
#define LCE_UNLIKELY(condition) (condition)
#endif

#if defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

// Check that 0 <= index < limit using a single comparison, assuming
// that 0 <= limit if Index is signed.  Intended for use in performance
// critical contexts where 0 <= index < limit is almost always true.
inline bool FastBoundsCheck(const int index, const int limit) {
  return LCE_LIKELY((unsigned)index < (unsigned)limit);
}

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

// Clamp an int32 value to int8 range
inline std::int8_t saturate(std::int32_t x) {
#ifdef __arm__
  std::int8_t y;
  asm("ssat %[y], #8, %[x]\n" : [y] "=r"(y) : [x] "r"(x));
  return y;
#else
  x = std::min<std::int32_t>(x, std::numeric_limits<std::int8_t>::max());
  x = std::max<std::int32_t>(x, std::numeric_limits<std::int8_t>::lowest());
  return static_cast<std::int8_t>(x);
#endif
}

// arithmetic right shift and clamp an int32 value to int8 range
template <int shift>
inline std::int8_t shift_saturate(std::int32_t x) {
#ifdef __arm__
  std::int8_t y;
  asm("ssat %[y], #8, %[x], asr %[shift]\n"
      : [y] "=r"(y)
      : [x] "r"(x), [shift] "i"(shift));
  return y;
#else
  x = x >> shift;
  x = std::min<std::int32_t>(x, std::numeric_limits<std::int8_t>::max());
  x = std::max<std::int32_t>(x, std::numeric_limits<std::int8_t>::lowest());
  return static_cast<std::int8_t>(x);
#endif
}

// Round-to-nearest. Handling of ties is allowed to be anything, as discussed in
// https://github.com/tensorflow/tensorflow/issues/25087
inline std::int32_t round(float x) {
#if defined(__thumb__) && defined(__VFP_FP__) && !defined(__SOFTFP__)
  // The `vcvtr` instructions follows the IEEE 754 rounding standard which
  // rounds halfway points to the nearest *even* integer.
  std::int32_t y;
  asm("vcvtr.s32.f32 %[x], %[x] \n"
      "vmov %[y], %[x] \n"
      : [y] "=r"(y)
      : [x] "t"(x));  // The "t" means `x` will be in an FPU register
  return y;
#else
  return ::tflite::TfLiteRound(x);
#endif
}

template <typename T, typename S>
constexpr T CeilDiv(T a, S b) {
  return (a + b - 1) / b;
}

template <typename T, typename S>
constexpr T Ceil(T a, S b) {
  return CeilDiv(a, b) * b;
}

}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_TYPES_H_
