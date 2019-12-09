#ifndef COMPUTE_ENGINE_AARCH64_PACKBITS_H
#define COMPUTE_ENGINE_AARCH64_PACKBITS_H

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <cstdint>

namespace compute_engine {
namespace core {

// This will bitpack exactly 64 floats.
// It will be packed in a weird order which is described in the comments
void packbits_aarch64_64(const float* input, uint64_t* output) {
  asm volatile(
      "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
      "sri    v0.4s, v2.4s, #1    \n"
      "sri    v1.4s, v3.4s, #1    \n"
      "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%0], #64    \n"
      "sri    v2.4s, v4.4s, #1    \n"
      "sri    v3.4s, v5.4s, #1    \n"
      "sri    v0.4s, v2.4s, #2    \n"
      "sri    v1.4s, v3.4s, #2    \n"
      // v0,v1 contain 4 signbits in each lane
      "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%0], #64    \n"
      "sri    v2.4s, v4.4s, #1    \n"
      "sri    v3.4s, v5.4s, #1    \n"
      "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"
      "sri    v4.4s, v6.4s, #1    \n"
      "sri    v5.4s, v7.4s, #1    \n"
      "sri    v2.4s, v4.4s, #2    \n"
      "sri    v3.4s, v5.4s, #2    \n"
      "sri    v0.4s, v2.4s, #4    \n"
      "sri    v1.4s, v3.4s, #4    \n"
      "sri    v0.4s, v1.4s, #8    \n"
      // v0-0 ... v0-3 all hold 16 signbits
      // Now its tricky because everything is already
      // inside the single SIMD register v0 now.
      // Now we will treat v0 as consisting of 8 times
      // a 16-bit number instead of 4 times a 32-bit number:
      // v0.h[0] -- garbage <--\
      // v0.h[1] -- signs      |
      // v0.h[2] -- garbage <--|--\
      // v0.h[3] -- signs      |  |
      // v0.h[4] -- garbage    |  |
      // v0.h[5] -- signs -----/  |
      // v0.h[6] -- garbage       |
      // v0.h[7] -- signs --------/
      "mov    v0.h[0], v0.h[5] \n"
      "mov    v0.h[2], v0.h[7] \n"
      // Flip all bits.
      // TODO: we can remove this instruction if we flip the zero-padding
      // correction
      "not    v0.8b, v0.8b   \n"
      // Store in output. %1 is the output pointer, post-increment it
      "st1    {v0.1d}, [%1], #8 \n"

      // clang-format off
      // Final result:
      //Bit i comes from input:
      //  62  46  30  14  54  38  22   6 |  58  42  26  10  50  34  18   2 |  60  44  28  12  52  36  20   4 |  56  40  24   8  48  32  16   0
      //  63  47  31  15  55  39  23   7 |  59  43  27  11  51  35  19   3 |  61  45  29  13  53  37  21   5 |  57  41  25   9  49  33  17   1
      //
      //Input i goes to bit:
      //  31  63  15  47  23  55   7  39 |  27  59  11  43  19  51   3  35 |  30  62  14  46  22  54   6  38 |  26  58  10  42  18  50   2  34
      //  29  61  13  45  21  53   5  37 |  25  57   9  41  17  49   1  33 |  28  60  12  44  20  52   4  36 |  24  56   8  40  16  48   0  32
      // clang-format on
      : "+r"(input),  // %0
        "+r"(output)  // %1
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
  return;
}

// For packing an entire array
void packbits_aarch64_64(const float* input, size_t num_blocks,
                         uint64_t* output) {
  while (num_blocks--) {
    packbits_aarch64_64(input, output++);
    input += 64;
  }
  return;
}

}  // namespace core
}  // namespace compute_engine

#endif
