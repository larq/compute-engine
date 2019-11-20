#ifndef COMPUTE_ENGINE_AARCH64_PACKBITS_H
#define COMPUTE_ENGINE_AARCH64_PACKBITS_H

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <cstdint>

//
// Benchmarking.
// WARNING: Make sure to disable cpu frequency scaling before testing these!
//      sudo cpupower frequency-set --governor performance
//      ./benchmark_aarch64
//      sudo cpupower frequency-set --governor powersave
//
// Time in nanoseconds
//                           128 |   512 |   64K |
//                          -----|-------|-------|
//  64bit  no `prfm`          53 |   210 | 28375 |
//  64bit  no prfm  bitflip   57 |   225 | 30400 |
//  64bit  pldl1strm before   51 |   210 | 28381 |
//  64bit  pldl1strm start    54 |   216 | 28850 |
//  64bit  pldl1keep start    55 |   215 | 28830 |
//  64bit  pldl1strm mid      53 |   210 | 28470 |
//  64bit  pldl1keep mid      53 |   210 | 28380 |
// 128bit  no `prfm`          53 |   210 | 28344 |
// 128bit  no prfm  bitflip   54 |   213 | 28342 |
// 128bit  pldl1strm before   54 |   210 | 28354 |
// 128bit  pldl1keep v1       53 |   210 | 28194 |
// 128bit  pldl1strm v1       53 |   210 | 28200 |
// 128bit  pldl1strm 8times   55 |   215 | 32092 |
// 128bit  pldl1strm 4times   54 |   210 | 35930 |
//  dabnn                     54 |   212 | 37600 |
//
// From these numbers, it looks as if the Raspberry Pi 4 CPU does not benefit
// from the prefetch instruction so we should leave it out.
// Furthermore, the extra bitflip has more impact in the 64-bit packing.

// General purpose registers:
// x0...x30 are 64-bit registers
// w0...w30 are the 32-bit bottom halves of the same registers

// SIMD registers:
//
// There are 32 SIMD registers: v0 ... v31.
// Each vN is 128-bit or 16 bytes (or 4 floats).
//
// You can use them in "vector" mode and "scalar" mode
//
// https://developer.arm.com/docs/den0024/latest/armv8-registers/neon-and-floating-point-registers/scalar-register-sizes
// Scalar mode only uses the register partially:
// q0 ... q31    128-bit (uses them completely)
// d0 ... d31     64-bit (uses half of the bits)
// s0 ... s31     32-bit (uses quarter of the bits)
// h0 ... h31     16-bit (...)
// b0 ... b31      8-bit (...)
//
// https://developer.arm.com/docs/den0024/a/armv8-registers/neon-and-floating-point-registers/vector-register-sizes
// Vector mode. v0 can be any vN register here:
// v0.4s      4 times 32-bit
// v0.2d      2 times 64-bit
// v0.16b    16 times  8-bit
// v0.2s      2 times 32-bit (others unused)
//
// If we want to access a single element of the vector
// register we can do so by writing v0.s[2] for example
//
// When writing to a register in scalar mode,
// for example writing to s7, then it will set
// the 96 remaining bits of v7 to zero.

// PDF with all the instructions
// https://static.docs.arm.com/ddi0596/a/DDI_0596_ARM_a64_instruction_set_architecture.pdf
//

// Prefetch memory instructions:
//  prfm pldl1keep, [pointer]
//  prfm pldl1strm, [pointer]
//
// Explanation:
// pld       - preLOAD instead of preSTORE
// l1        - to L1 cache
// keep/strm - "keep in cache policy" or "streaming data policy"
//
// The keep policy seems to be if we want to keep reading/writing to this.
// The stream policy is meant for data that is only read once, for example
// by functions like memcpy. This suggests we should also use strm.

namespace compute_engine {
namespace core {

// Output should be allocated by caller
// It will pack exactly num_blocks times 128 bits
// num_blocks needs to be >= 1 or it will crash
// Every block will be packed in a weird order which is described in the
// comments at the end
void packbits_aarch64_128(const float* input, size_t num_blocks, void* output) {
  // This is based on daBNN bitpacking for Aarch64
  // https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h
  asm volatile(
      //"prfm   pldl1strm, [%0]     \n"
      "0: \n"
      // Load memory into registers
      // http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0802a/LD1_advsimd_sngl_vector.html
      //
      // We are loading into 4 128-bit registers with one instruction
      // so we loaded 16 floats = 64 bytes.
      // The #64 at the end means increment the pointer by 64 bytes
      // after reading the memory, like a `y = *x++` instruction.
      // Note: this instruction only supports consecutive registers
      // so you can not do v0, v3, v1, v2 or something.
      "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
      "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"
      // Checkpoint1

      // See the PDF for the SRI instruction
      // sri destination, source, number of bits
      //
      // It shifts source to the right by the specified number of bits
      // and then puts the result in destination, but importantly
      // it *keeps the upper bits of destination intact*.
      //
      // So if we have four floats:
      // s1: signbit1 otherbits1
      // s2: signbit2 otherbits2
      // s3: signbit3 otherbits3
      // s4: signbit4 otherbits4
      //
      // Now if we would do (unvectorized)
      //                    Contents of s0 afterwards:
      // sri s0, s1, 0      signbit1 otherbits1
      // sri s0, s2, 1      signbit1 signbit2 otherbits2(-1)
      // sri s0, s3, 2      signbit1 signbit2 signbit3 otherbits3(-2)
      // sri s0, s4, 3      signbit1 signbit2 signbit3 signbit4 otherbits4(-3)
      //
      // We can do the same on the vectorized registers:
      "sri    v0.4s, v4.4s, #1    \n"
      "sri    v1.4s, v5.4s, #1    \n"
      "sri    v2.4s, v6.4s, #1    \n"
      "sri    v3.4s, v7.4s, #1    \n"
      // Checkpoint2
      // The four 32-bit parts in v0 contain the signbits of v0 and of v4
      // The four 32-bit parts in v1 contain the signbits of v1 and of v5
      // ...
      // The registers v4...v7 are free to use again
      "ld1    {v4.4s, v5.4s,  v6.4s,  v7.4s}, [%0], #64    \n"
      "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      // Checkpoint3
      "sri    v4.4s,  v8.4s, #1    \n"
      "sri    v5.4s,  v9.4s, #1    \n"
      "sri    v6.4s, v10.4s, #1    \n"
      "sri    v7.4s, v11.4s, #1    \n"
      // Checkpoint4
      // Now v4...v7 contain 2 signbits in each 32-bit part
      // v8...v11 are free to use
      "sri    v0.4s,  v4.4s, #2    \n"
      "sri    v1.4s,  v5.4s, #2    \n"
      "sri    v2.4s,  v6.4s, #2    \n"
      "sri    v3.4s,  v7.4s, #2    \n"
      // Checkpoint5
      // Now v0...v3 contain 4 signbits in each 32-bit part
      // v4...v11 are all free to use
      // Do the same thing but starting at v4
      "ld1    {v4.4s, v5.4s,  v6.4s,  v7.4s}, [%0], #64    \n"
      "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      // Checkpoint6
      "sri    v4.4s,  v8.4s, #1    \n"
      "sri    v5.4s,  v9.4s, #1    \n"
      "sri    v6.4s, v10.4s, #1    \n"
      "sri    v7.4s, v11.4s, #1    \n"
      // Checkpoint7
      // v0...v3 contain 4 signbits in each 32-bit part
      // v4...v7 contain 2 signbits in each 32-bit part
      "ld1    { v8.4s,  v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64    \n"
      // Checkpoint8

      // From this point on we only do `sri` so lets try
      // to preload things for the next loop while we do the computations.
      // It is completely unclear how many bytes are prefetched by this
      // instruction. It might be one cache line which is 64 bytes on the
      // Cortex-A72. Since we read 8 cache lines in total, lets try to call it 8
      // times.
      //"prfm   pldl1strm, [%0]     \n"
      //"prfm   pldl1strm, [%0, 64] \n"
      //"prfm   pldl1strm, [%0,128] \n"
      //"prfm   pldl1strm, [%0,192] \n"
      //"prfm   pldl1strm, [%0,256] \n"
      //"prfm   pldl1strm, [%0,320] \n"
      //"prfm   pldl1strm, [%0,384] \n"
      //"prfm   pldl1strm, [%0,448] \n"

      "sri     v8.4s, v12.4s, #1    \n"
      "sri     v9.4s, v13.4s, #1    \n"
      "sri    v10.4s, v14.4s, #1    \n"
      "sri    v11.4s, v15.4s, #1    \n"
      // v8...v11 contain 2 signbits in each 32-bit part
      "sri    v4.4s,  v8.4s, #2    \n"
      "sri    v5.4s,  v9.4s, #2    \n"
      "sri    v6.4s, v10.4s, #2    \n"
      "sri    v7.4s, v11.4s, #2    \n"
      // Checkpoint9
      // v4...v7 contain 4 signbits in each 32-bit part
      "sri    v0.4s,  v4.4s, #4    \n"
      "sri    v1.4s,  v5.4s, #4    \n"
      "sri    v2.4s,  v6.4s, #4    \n"
      "sri    v3.4s,  v7.4s, #4    \n"
      // Checkpoint10
      // v0...v3 contain 8 signbits in each 32-bit part
      "sri    v0.4s,  v1.4s, #8    \n"
      "sri    v2.4s,  v3.4s, #8    \n"
      // v0 and v2 contain 16 signbits in each 32-bit part
      "sri    v0.4s,  v2.4s, #16   \n"
      // v0 contains 32 signbits in each 32-bit part
      // So in total v0 now has 128 signbits!

      // clang-format off
      // Timesteps in horizontal direction
      // C0 stands for Checkpoint0
      // A dash - means the register was unchanged
      // The signbits are always in the most-significant-bits of the register.
      //       C1       C2       C3       C4       C5          C6       C7       C8       C9       C10
      // Reg   Signbits Signbits Signbits Signbits Signbits    Signbits Signbits Signbits Signbits Signbits
      // v0-0  0        0,16     -        -        0,16,32,48  -        -        -        -        0,16,32,48,64,80,96,112
      // v0-1  1        1,17     -        -        1,17,33,49  -        -        -        -        1,17,33,49,65,81,97,113
      // v0-2  2        2,18     -        -        2,18,34,50  -        -        -        -        2,18,...,114
      // v0-3  3        3,19     -        -        3,19,35,51  -        -        -        -        3,19,...,115
      // v1-0  4        4,20     -        -        4,20,36,52  -        -        -        -        4,20,...,116
      // ...
      // v2-0  8        8,24     -        -        8,24,40,56  -        -        -        -        4,20,...,116
      // ...
      // v3-2  14       14,30    -        -        14,30,46,62 -        -        -        -        14,30,46,62,...,126
      // v3-3  15       15,31    -        -        15,31,47,63 -        -        -        -        15,31,47,63,...,127
      // v4-0  16       -        32       32,48    -           64       64,80    -        64,80,96,112
      // v4-1  17       -        33       33,49    -           65       65,81    -        65,81,97,113
      // ...
      // v7-2  30       -        46       46,62    -           78       78,94    -        78,94,110,126
      // v7-3  31       -        47       47,63    -           79       79,95    -        79,95,111,127
      // v8-0  -        -        48       -        -           80       -        96       -
      // ...
      // v11-3 -        -        63       -        -           95       -        111      -
      // ...
      // v15-3 -        -        -        -        -           -        -        127      -
      //
      // In the end we get
      //  0, 16, 32, 48, 64, 80, 96,112,
      //  4, 20, 36, 52, 68, 84,100,116,
      //  8, 24, 40, 56, 72, 88,104,120,
      // 12, 28, 44, 60, 76, 92,108,124,
      //  1, 17, 33, 49, 65, 81, 97,113,
      //  5, 21, ...
      //  9, 25, ...
      // 13, 29, ...
      //  2, ..
      //  6, ..
      // 10, ..
      // 14, ..
      //  3, ..
      //  7, ..
      // 11, ..
      // 15, ..., 127
      //
      // However, for every group of 32-bits we have to reverse the order now because we started storing them at the most-significant-bit
      // The final result, therefore, is as follows, generated by a unittest:
      // 
      //Bit i comes from input:
      // 124 108  92  76  60  44  28  12 | 120 104  88  72  56  40  24   8 | 116 100  84  68  52  36  20   4 | 112  96  80  64  48  32  16   0
      // 125 109  93  77  61  45  29  13 | 121 105  89  73  57  41  25   9 | 117 101  85  69  53  37  21   5 | 113  97  81  65  49  33  17   1
      // 126 110  94  78  62  46  30  14 | 122 106  90  74  58  42  26  10 | 118 102  86  70  54  38  22   6 | 114  98  82  66  50  34  18   2
      // 127 111  95  79  63  47  31  15 | 123 107  91  75  59  43  27  11 | 119 103  87  71  55  39  23   7 | 115  99  83  67  51  35  19   3
      //
      //The other way around: input i goes to bit:
      //  31  63  95 127  23  55  87 119 |  15  47  79 111   7  39  71 103 |  30  62  94 126  22  54  86 118 |  14  46  78 110   6  38  70 102
      //  29  61  93 125  21  53  85 117 |  13  45  77 109   5  37  69 101 |  28  60  92 124  20  52  84 116 |  12  44  76 108   4  36  68 100
      //  27  59  91 123  19  51  83 115 |  11  43  75 107   3  35  67  99 |  26  58  90 122  18  50  82 114 |  10  42  74 106   2  34  66  98
      //  25  57  89 121  17  49  81 113 |   9  41  73 105   1  33  65  97 |  24  56  88 120  16  48  80 112 |   8  40  72 104   0  32  64  96
      // clang-format on

      // Flip all bits: TODO: flip bgemm instead
      "not    v0.16b, v0.16b        \n"
      // Store in output. %1 is the output pointer, increment it by 16 bytes
      "st1    {v0.4s}, [%1], #16    \n"

      // Subtract the packed_elements counter by one
      "subs   %2, %2, #1          \n"
      // Jump back to start if not equal to 0
      // (label 0, b for backwards)
      "bne    0b                  \n"
      : "+r"(input),      // %0
        "+r"(output),     // %1
        "+r"(num_blocks)  // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  // "cc"     -- we change 'processor flags'
  // "memory" -- we read/write memory pointed to by the input/output variables
  return;
}

// Same as above but in blocks of size 64
void packbits_aarch64_64(const float* input, size_t num_blocks, void* output) {
  asm volatile(
      //"prfm   pldl1strm, [%0]     \n"
      "0: \n"
      "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
      "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"
      "sri    v0.4s, v4.4s, #1    \n"
      "sri    v1.4s, v5.4s, #1    \n"
      "sri    v2.4s, v6.4s, #1    \n"
      "sri    v3.4s, v7.4s, #1    \n"
      // The four 32-bit parts in v0 contain the signbits of v0 and of v4
      // The four 32-bit parts in v1 contain the signbits of v1 and of v5
      // ...
      // The registers v4...v7 are free to use again
      "ld1    {v4.4s, v5.4s,  v6.4s,  v7.4s}, [%0], #64    \n"
      "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      "sri    v4.4s,  v8.4s, #1    \n"
      "sri    v5.4s,  v9.4s, #1    \n"
      "sri    v6.4s, v10.4s, #1    \n"
      "sri    v7.4s, v11.4s, #1    \n"
      // Now v4...v7 contain 2 signbits in each 32-bit part
      "sri    v0.4s,  v4.4s, #2    \n"
      "sri    v1.4s,  v5.4s, #2    \n"
      "sri    v2.4s,  v6.4s, #2    \n"
      "sri    v3.4s,  v7.4s, #2    \n"
      // v0-0 holds 4 signbits
      // v0-1 holds 4 signbits
      // ...
      // v3-3 holds 4 signbits
      // So that is 16 * 4 = 64 signbits in total.
      // Now we simpy have to combine them!
      "sri    v0.4s,  v2.4s, #4    \n"
      "sri    v1.4s,  v3.4s, #4    \n"
      // v0-0 ... v1-3 all hold 8 signbits
      "sri    v0.4s,  v1.4s, #8    \n"
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
      // Flip all bits: TODO: flip bgemm instead
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

      // Subtract the packed_elements counter by one
      "subs   %2, %2, #1          \n"
      // Jump back to start if not equal to 0
      // (label 0, b for backwards)
      "bne    0b                  \n"
      : "+r"(input),      // %0
        "+r"(output),     // %1
        "+r"(num_blocks)  // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  // "cc"     -- we change 'processor flags'
  // "memory" -- we read/write memory pointed to by the input/output variables
  return;
}

}  // namespace core

namespace benchmarking {

//
// The ones below are for benchmarking only
//
// Ordering:
// The least significant bit of out (i.e.  x & 0x01) should be the sign of the
// *first* input

// Benchmarks performed on Raspberry Pi 4 (Cortex-A72)

// From the manual at
// https://developer.arm.com/docs/100069/0607/a64-general-instructions/bfm
// BitfieldInsert copies any number of low-order bits from a source register
// into the same number of adjacent bits at any position in the destination
// register, leaving other bits unchanged.

//
// Version 1
// 52000 ns on 32K floats
// Uses 6 instructions per float
uint64_t pack64bits_v1(float* in) {
  uint32_t* in_ptr = reinterpret_cast<uint32_t*>(in);
  uint64_t x = 0;
  for (auto j = 0u; j < 64; ++j) {
    x <<= 1;
    uint32_t y = *in_ptr++;
    x |= y >> 31;
  }
  // Reverse the bits in x
  asm("rbit %0, %0\n" : "+r"(x) : :);
  return x;
  // Compiled with -O3, the above loop becomes:
  //.L4:
  //	ldr	w0, [x1], 4         y = *in_ptr++;
  //	cmp	x1, x3              compare  in_ptr < end_in_ptr
  //	lsr	w0, w0, 31          y >>= 31
  //	bfi	x0, x2, 1, 63       y[1..64] = x[0..63] (BitFieldInsert)
  //	mov	x2, x0              x = y
  //	bne	.L4                 if compare was true, jump back
  // Note that w0 is the lower 32-bit part of the 64-bit x0 register
}

//
// Version 2
// 70000 ns on 32K floats
uint64_t pack64bits_v2(float* in) {
  uint32_t* in_ptr = reinterpret_cast<uint32_t*>(in);
  uint64_t x = 0;
  long int j = 31;
  while (j >= 0) {
    uint32_t y = *in_ptr++;
    uint64_t y_bit = y & (1u << 31);
    // y_bit has the correct bit at bit 31
    // so shift it down to get the bit at 0, 1, 2, ...
    x |= (y_bit >> j);
    j--;
  }
  for (auto j = 1u; j <= 32; ++j) {
    uint32_t y = *in_ptr++;
    uint64_t y_bit = y & (1u << 31);
    // y_bit has the correct bit at 31
    // shift it up to get the bit at 32, 33, 34, ...
    x |= (y_bit << j);
  }
  return x;
  // Compiled with -O3, the above loop becomes:
  //.L6:
  //	ldr	w1, [x3], 4
  //	and	w1, w1, -2147483648
  //	lsr	x1, x1, x2
  //	sub	x2, x2, #1
  //	orr	x0, x0, x1
  //	cmn	x2, #1
  //	bne	.L6
  //
  //	add	x3, x4, 128
  //	mov	x2, 0
  //.L7:
  //	ldr	w1, [x3, x2, lsl 2]
  //	and	w1, w1, -2147483648
  //	lsl	x1, x1, x2
  //	add	x2, x2, 1
  //	orr	x0, x0, x1
  //	cmp	x2, 32
  //	bne	.L7
}

//
// Version 3
// 70000 ns on 32K floats
// Uses 6 instructions per float
uint64_t pack64bits_v3(float* in) {
  float* in_end = &in[64];
  uint64_t x = 0;
  asm("0:\n"
      "lsl %0, %0, #1       \n"  // x <<= 1
      "ldr w1, [%1], 4      \n"  // y = *in++;
      "lsr w1, w1, 31       \n"  // y = (y >> 31)
      "bfi %0, x1, #0, #1   \n"  // x[0] = w0[0]
      "cmp %1, %2           \n"  // if  in < in_end
      "bne 0b               \n"  // then loop back
      "rbit %0, %0          \n"  // reverse x
      : "+r"(x),                 // %0
        "+r"(in),                // %1
        "+r"(in_end)             // %2
      :
      : "cc", "memory", "w1");
  return x;
}

//
// Version 4
// 56000 ns on 32K floats
// Uses 4 instructions per float
uint64_t pack64bits_v4(float* in) {
  float* in_end = &in[32];
  uint64_t x = 0;
  asm("ldr w1, [%1], 4      \n"  // w1 = *in++;
      "lsr w1, w1, 1        \n"  // w1 = w1 >> 1
      "ldr w2, [%1], 4      \n"  // w2 = *in++;
      "bfi w2, w1, #0, #31  \n"  // w2[0:30] = w1[0:30]
      "1:\n"
      "lsr w2, w2, 1        \n"  // w2 = (w2 >> 1)
      "ldr w1, [%1], 4      \n"  // w1 = *in++;
      "bfi w1, w2, #0, #31  \n"  // w1[0:30] = w2[0:30]
      "lsr w1, w1, 1        \n"  // w1 = (w1 >> 1)
      "ldr w2, [%1], 4      \n"  // w1 = *in++;
      "bfi w2, w1, #0, #31  \n"  // w2[0:30] = w1[0:30]
      "cmp %1, %2           \n"  // if  in < in_end
      "bne 1b               \n"  // then loop back
      "mov %0, x2           \n"  // save in x
      // Second set of 32 bits
      "add %2, %2, 128      \n"  // in_end += 32*4
      "ldr w1, [%1], 4      \n"  // w1 = *in++;
      "lsr w1, w1, 1        \n"  // w1 = w1 >> 1
      "ldr w2, [%1], 4      \n"  // w2 = *in++;
      "bfi w2, w1, #0, #31  \n"  // w2[0:30] = w1[0:30]
      "2:\n"
      "lsr w2, w2, 1        \n"  // w2 = (w2 >> 1)
      "ldr w1, [%1], 4      \n"  // w1 = *in++;
      "bfi w1, w2, #0, #31  \n"  // w1[0:30] = w2[0:30]
      "lsr w1, w1, 1        \n"  // w1 = (w1 >> 1)
      "ldr w2, [%1], 4      \n"  // w1 = *in++;
      "bfi w2, w1, #0, #31  \n"  // w2[0:30] = w1[0:30]
      "cmp %1, %2           \n"  // if  in < in_end
      "bne 2b               \n"  // then loop back
      // now %0 contains the correct lower 32 bits
      // and w2 contains the 32 bits that should go into the higher part of %0
      "orr %0, %0, x2, lsl 32 \n"
      : "+r"(x),      // %0
        "+r"(in),     // %1
        "+r"(in_end)  // %2
      :
      : "cc", "memory", "w1", "w2");
  return x;
}

//
// Version 5
// 57000 ns on 32K floats
// Uses 3 instructions per float
uint64_t pack64bits_v5(float* in) {
  float* in_end = &in[32];
  uint64_t x = 0;
  asm("mov w2, 0            \n"
      "1:\n"
      "lsr w2, w2, 1        \n"
      "ldp w1, w3, [%1], 8  \n"  // load pair
      "bfi w1, w2, #0, #31  \n"
      "lsr w1, w1, 1        \n"
      "bfi w3, w1, #0, #31  \n"
      "lsr w3, w3, 1        \n"
      "ldp w1, w2, [%1], 8  \n"
      "bfi w1, w3, #0, #31  \n"
      "lsr w1, w1, 1        \n"
      "bfi w2, w1, #0, #31  \n"
      "cmp %1, %2           \n"
      "bne 1b               \n"
      "mov %0, x2           \n"
      // Second set of 32 bits
      "add %2, %2, 128      \n"
      "mov w2, 0            \n"
      "2:\n"
      "lsr w2, w2, 1        \n"
      "ldp w1, w3, [%1], 8  \n"
      "bfi w1, w2, #0, #31  \n"
      "lsr w1, w1, 1        \n"
      "bfi w3, w1, #0, #31  \n"
      "lsr w3, w3, 1        \n"
      "ldp w1, w2, [%1], 8  \n"
      "bfi w1, w3, #0, #31  \n"
      "lsr w1, w1, 1        \n"
      "bfi w2, w1, #0, #31  \n"
      "cmp %1, %2           \n"
      "bne 2b               \n"
      // now %0 contains the correct lower 32 bits
      // and w2 contains the 32 bits that should go into the higher part of %0
      "orr %0, %0, x2, lsl 32 \n"
      : "+r"(x),      // %0
        "+r"(in),     // %1
        "+r"(in_end)  // %2
      :
      : "cc", "memory", "w1", "w2", "w3");
  return x;
}

// This is a direct copy of the daBNN code, only used for comparison.
void packbits_dabnn_128(const float* float_ptr, size_t nn_size,
                        void* binary_ptr) {
  asm volatile(
      "0:     \n"
      "prfm   pldl1keep, [%0]     \n"
      "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
      "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"
      "sri    v0.4s, v4.4s, #1    \n"
      "sri    v1.4s, v5.4s, #1    \n"
      "sri    v2.4s, v6.4s, #1    \n"
      "sri    v3.4s, v7.4s, #1    \n"

      "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      "prfm   pldl1keep, [%0, #64]     \n"
      "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64    \n"
      "sri    v8.4s, v12.4s, #1    \n"
      "sri    v9.4s, v13.4s, #1    \n"
      "sri    v10.4s, v14.4s, #1    \n"
      "sri    v11.4s, v15.4s, #1    \n"

      "subs   %2, %2, #1          \n"

      "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64    \n"
      "prfm   pldl1keep, [%0, #64]     \n"
      "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64    \n"

      "sri    v0.4s, v8.4s, #2    \n"
      "sri    v1.4s, v9.4s, #2    \n"
      "sri    v2.4s, v10.4s, #2   \n"
      "sri    v3.4s, v11.4s, #2   \n"

      "sri    v16.4s, v20.4s, #1    \n"
      "sri    v17.4s, v21.4s, #1    \n"
      "sri    v18.4s, v22.4s, #1    \n"
      "sri    v19.4s, v23.4s, #1    \n"

      "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
      "prfm   pldl1keep, [%0, #64]     \n"
      "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64    \n"
      "sri    v8.4s, v12.4s, #1    \n"
      "sri    v9.4s, v13.4s, #1    \n"
      "sri    v10.4s, v14.4s, #1    \n"
      "sri    v11.4s, v15.4s, #1    \n"

      "sri    v16.4s, v8.4s, #2   \n"
      "sri    v17.4s, v9.4s, #2   \n"
      "sri    v18.4s, v10.4s, #2   \n"
      "sri    v19.4s, v11.4s, #2   \n"

      "sri    v0.4s, v16.4s, #4   \n"
      "sri    v1.4s, v17.4s, #4   \n"
      "sri    v2.4s, v18.4s, #4   \n"
      "sri    v3.4s, v19.4s, #4   \n"

      "sri    v0.4s, v1.4s, #8    \n"
      "sri    v2.4s, v3.4s, #8    \n"
      "sri    v0.4s, v2.4s, #16    \n"

      "not    v0.16b, v0.16b        \n"

      "st1    {v0.4s}, [%1], #16         \n"
      "bne    0b                  \n"
      : "+r"(float_ptr),   // %0
        "+r"(binary_ptr),  // %1
        "+r"(nn_size)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "x0");
}

}  // namespace benchmarking

}  // namespace compute_engine

#endif
