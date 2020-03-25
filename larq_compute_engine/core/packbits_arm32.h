#ifndef COMPUTE_ENGINE_ARM32_PACKBITS_H
#define COMPUTE_ENGINE_ARM32_PACKBITS_H

namespace compute_engine {
namespace core {

// They discuss the vectorized `vsri` for arm32 here:
// https://stackoverflow.com/questions/49918746/efficiently-accumulate-sign-bits-in-arm-neon

// Output should be allocated by caller
// It should have size equal to (n+63)/64
void packbits_arm32(const float* input, const std::size_t n,
                    std::uint64_t* output) {
  //
  // This is *not* yet valid bitpacking code.
  //
  // It is just a test to see if ARM assembly works with our
  // bazel compilation and qemu setup.
  //
  // It computes output[0] = input[0] ^ input[1]
  // Interpreted as if output and input are both uint32_t
  //

  std::size_t packed_elements = (n + 63) / 64;
  asm volatile(
      // Prefetch memory. %0 is input pointer
      "pld [%0] \n"
      // Load into r0 register and increment input pointer by 4 bytes
      "ldr r0, [%0], #4 \n"
      // Load into r1
      "ldr r1, [%0], #4 \n"
      // r0 = (r0 XOR r1)
      "eor r0, r0, r1 \n"
      // Store result in output pointer
      "str r0, [%1] \n"

      // Subtract packed_elements by one
      "subs %2, %2, #1 \n"

      : "+r"(input),           // %0
        "+r"(output),          // %1
        "+r"(packed_elements)  // %2
      :
      : "r0", "r1");
  return;
}
}  // namespace core
}  // namespace compute_engine

#endif
