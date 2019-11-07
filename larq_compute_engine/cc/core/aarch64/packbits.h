#ifndef COMPUTE_ENGINE_AARCH64_PACKBITS_H
#define COMPUTE_ENGINE_AARCH64_PACKBITS_H

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

namespace compute_engine {
namespace core {

// TODO: Checkout daBNN bitpacking
// https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h
// They seem to use un-vectorized `sri` instructions ??
// They discuss the vectorized `vsri` here:
// https://stackoverflow.com/questions/49918746/efficiently-accumulate-sign-bits-in-arm-neon
//

// Output should be allocated by caller
// It should have size equal to (n+63)/64
void packbits_aarch64(const float* input, const size_t n, uint64_t* output) {
  //
  // This is *not* yet valid bitpacking code.
  //
  // It is just a test to see if ARM assembly works with our
  // bazel compilation and qemu setup.
  //

  size_t packed_elements = (n + 63) / 64;
  asm volatile(
      // Prefetch memory.
      // %0 is the input pointer
      // pld  - preLOAD instead of preSTORE
      // l1   - to L1 cache
      // keep - "keep in cache policy". what does this mean???
      "prfm   pldl1keep, [%0]     \n"

      // Load memory into registers
      // http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0802a/LD1_advsimd_sngl_vector.html
      //
      // v0.4s means:
      // - register v0, this is a 128-bit register (4 floats; 16 bytes)
      // - 4s means 4 times 32-bit
      // We are loading into 4 such 128-bit registers with one instruction
      // so we loaded 16 floats = 64 bytes.
      // The #64 at the end means increment the pointer by 64 bytes
      // after reading the memory, like a `y = *x++` instruction.
      "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
      "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"

      // Shift right to get the sign bit and accumulate
      "sri    v0.4s, v4.4s, #1    \n"
      "sri    v1.4s, v5.4s, #1    \n"
      "sri    v2.4s, v6.4s, #1    \n"
      "sri    v3.4s, v7.4s, #1    \n"

      // Subtract the packed_elements counter by one
      "subs   %2, %2, #1          \n"

      // TODO: loop back and so on

      // Store in output. %1 is the output pointer
      "st1    {v0.4s}, [%1], #16         \n"

      : "+r"(input),           // %0
        "+r"(output),          // %1
        "+r"(packed_elements)  // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "x0");
  return;
}
}  // namespace core
}  // namespace compute_engine

#endif
