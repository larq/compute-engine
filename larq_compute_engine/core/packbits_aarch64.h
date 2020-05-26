#ifndef COMPUTE_ENGINE_AARCH64_PACKBITS_H
#define COMPUTE_ENGINE_AARCH64_PACKBITS_H

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <cstdint>

namespace compute_engine {
namespace core {

// This will bitpack the sign-bits from 64 floats, in a deterministic but
// non-consecutive order.
inline void packbits_aarch64_64(const float* input, std::uint64_t* output) {
  asm volatile(
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[in_ptr]], #64\n"
      "sri v0.4s, v1.4s, #1\n"
      "ld1 {v4.4s, v5.4s}, [%[in_ptr]], #32\n"
      "sri v2.4s, v3.4s, #1\n"
      "ld1 {v6.4s, v7.4s}, [%[in_ptr]], #32\n"
      "sri v4.4s, v5.4s, #1\n"
      "ld1 {v8.4s, v9.4s}, [%[in_ptr]], #32\n"
      "sri v6.4s, v7.4s, #1\n"
      "ld1 {v10.4s, v11.4s}, [%[in_ptr]], #32\n"
      "sri v8.4s, v9.4s, #1\n"
      "ld1 {v12.4s, v13.4s}, [%[in_ptr]], #32\n"
      "sri v10.4s, v11.4s, #1\n"
      "ld1 {v14.4s, v15.4s}, [%[in_ptr]], #32\n"
      "sri v12.4s, v13.4s, #1\n"
      "sri v14.4s, v15.4s, #1\n"
      "sri v0.4s, v2.4s, #2\n"
      "sri v4.4s, v6.4s, #2\n"
      "sri v8.4s, v10.4s, #2\n"
      "sri v12.4s, v14.4s, #2\n"
      "sri v0.4s, v4.4s, #4\n"
      "sri v8.4s, v12.4s, #4\n"
      "sri v0.4s, v8.4s, #8\n"
      // v0.s[0] ... v0.s[3] all hold 16 signbits in the high half, and 16 bits
      // of junk data in the low half.
      //     With a 'unsigned saturating shift right and narrow' instruction, we
      // can collect all of the sign bits in v0.d[0].
      "uqshrn v0.4h, v0.4s, #16\n"
      "st1 {v0.1d}, [%[out_ptr]]\n"
      : [ in_ptr ] "+r"(input), [ out_ptr ] "+r"(output)
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  return;
}

// For packing an entire array
inline void packbits_aarch64_64(const float* input, std::size_t num_blocks,
                                std::uint64_t* output) {
  while (num_blocks--) {
    packbits_aarch64_64(input, output++);
    input += 64;
  }
  return;
}

}  // namespace core
}  // namespace compute_engine

#endif
