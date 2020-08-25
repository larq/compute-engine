#ifndef COMPUTE_ENGINE_AARCH64_BITPACK_H
#define COMPUTE_ENGINE_AARCH64_BITPACK_H

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <cstdint>

#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"

namespace compute_engine {
namespace core {

// Bitpack an array of `4 * 32 * num_blocks` floats.
inline void bitpack_aarch64_4x32(const float* input, std::size_t num_blocks,
                                 TBitpacked* output) {
  static_assert(sizeof(TBitpacked) == 4,
                "Correctness of this function relies on the size of TBitpacked "
                "being 4 bytes.");

  ruy::profiler::ScopeLabel label("Bitpack 4x32 floats (optimised)");

  if (num_blocks < 1) return;

  asm volatile(
      // Load 128 floats into the registers.
      //
      // These LD2 instructions load a sequence of pairs of two-byte half-words
      // from memory and de-interleave the pairs into two vector registers. This
      // means that the first vector register receives every even half-word, and
      // the second vector register receives every odd half-word.
      //     This is useful, because within every four-byte word we only care
      // about a single bit (the sign bit), and that bit is inside the odd
      // half-word of the pair. Thus, the de-interleaving done by the load
      // instruction has 'concentrated' the sign bits from 8 floats into a
      // single register rather than spread them evenly across both, which
      // reduces the amount of computation that we need to do.
      //     Note that we could use LD4 instructions for data loading instead
      // and de-interleave into bytes rather than half-words, which would
      // concentrate the sign-bits from 16 floats into a single register.
      // However, bit-packing is memory, not compute bound, and on some recent
      // Aarch64 CPUs (such as the Cortex A76) the execution throughput (and
      // latency) of using a sequence of LD4 instructions is significantly
      // higher than that of using a sequence of LD2 instructions.
      "ld2 {v0.8h, v1.8h}, [%[in_ptr]], #32\n"
      "ld2 {v2.8h, v3.8h}, [%[in_ptr]], #32\n"
      "ld2 {v4.8h, v5.8h}, [%[in_ptr]], #32\n"
      "ld2 {v6.8h, v7.8h}, [%[in_ptr]], #32\n"
      "ld2 {v8.8h, v9.8h}, [%[in_ptr]], #32\n"
      "ld2 {v10.8h, v11.8h}, [%[in_ptr]], #32\n"
      "ld2 {v12.8h, v13.8h}, [%[in_ptr]], #32\n"
      "ld2 {v14.8h, v15.8h}, [%[in_ptr]], #32\n"
      "cmp %[n], #1\n"
      "ld2 {v16.8h, v17.8h}, [%[in_ptr]], #32\n"
      "ld2 {v18.8h, v19.8h}, [%[in_ptr]], #32\n"
      "ld2 {v20.8h, v21.8h}, [%[in_ptr]], #32\n"
      "ld2 {v22.8h, v23.8h}, [%[in_ptr]], #32\n"
      "ld2 {v24.8h, v25.8h}, [%[in_ptr]], #32\n"
      "ld2 {v26.8h, v27.8h}, [%[in_ptr]], #32\n"
      "ld2 {v28.8h, v29.8h}, [%[in_ptr]], #32\n"
      "ld2 {v30.8h, v31.8h}, [%[in_ptr]], #32\n"
      "beq 1f\n"

      "2:\n"

      // This is the main loop.
      //
      // Here, n >= 2. We compute four int32 bitpacked words (from 128 floats)
      // while reloading data into the registers for the next iteration.
      //
      // After the LD2 instructions, each odd register v1, v3, ..., v31 contains
      // 8 sign bits, in the high-bit of each half-word. Correct order is
      // preserved, in the sense that v1 contains the first 8 sign bits, et
      // cetera, but also that the sign bits are correctly ordered within each
      // register.
      //     The first computation step is a series of UZP2 instructions on
      // pairs of registers that hold adjacent sign-bits. Every 'odd' byte is
      // taken from the two registers and concatenated together, preserving
      // order, in the destination register.
      //     After this block, the register sequence v16, v18, ..., v30 will
      // contain the sign bits (in correct order). Because we take every 'odd'
      // byte from each pair, we are effectively concentrating the sign bits, so
      // there will be 16 in each register, in the high-bit of each byte.
      "uzp2 v16.16b, v1.16b, v3.16b\n"
      "ld2 {v0.8h, v1.8h}, [%[in_ptr]], #32\n"
      "uzp2 v18.16b, v5.16b, v7.16b\n"
      "uzp2 v20.16b, v9.16b, v11.16b\n"
      "ld2 {v2.8h, v3.8h}, [%[in_ptr]], #32\n"
      "uzp2 v22.16b, v13.16b, v15.16b\n"
      "uzp2 v24.16b, v17.16b, v19.16b\n"
      "ld2 {v4.8h, v5.8h}, [%[in_ptr]], #32\n"
      "uzp2 v26.16b, v21.16b, v23.16b\n"
      "uzp2 v28.16b, v25.16b, v27.16b\n"
      "ld2 {v6.8h, v7.8h}, [%[in_ptr]], #32\n"
      "uzp2 v30.16b, v29.16b, v31.16b\n"

      // The next step is to perform two unzips of each adjacent pair of
      // sign-bit registers. The first (UZP2) takes 'odd' bytes as before, while
      // the second (UZP1) takes 'even' bytes.
      //
      // To illustrate:
      //
      // v16 and v18 contain the labelled bytes as follows (where each byte
      // contains a single sign-bit):
      //                        v16          v18
      //                   [0 1 ... 15] [16 17 ... 31]
      //
      // Afterwards, v0 and v17 will contain:
      //                        v0           v17
      //                   [1 3 ... 31] [0 2 ... 30]
      "uzp2 v0.16b, v16.16b, v18.16b\n"
      "ld2 {v8.8h, v9.8h}, [%[in_ptr]], #32\n"
      "uzp1 v17.16b, v16.16b, v18.16b\n"
      "uzp2 v2.16b, v20.16b, v22.16b\n"
      "ld2 {v10.8h, v11.8h}, [%[in_ptr]], #32\n"
      "uzp1 v19.16b, v20.16b, v22.16b\n"
      "uzp2 v4.16b, v24.16b, v26.16b\n"
      "ld2 {v12.8h, v13.8h}, [%[in_ptr]], #32\n"
      "uzp1 v21.16b, v24.16b, v26.16b\n"
      "uzp2 v6.16b, v28.16b, v30.16b\n"
      "ld2 {v14.8h, v15.8h}, [%[in_ptr]], #32\n"
      "uzp1 v23.16b, v28.16b, v30.16b\n"

      "sub %[n], %[n], #1\n"

      // Now, we can combine the two unzip results for each register pair, with
      // 'shift right and insert' SRI instructions.
      //
      // Continuing with the example from above, after the first SRI instruction
      // below, v0 will contain:
      //                                v0
      //                     [0/1  2/3  4/5 ... 30/31]
      //
      // Where `0/1` is a byte that has the high-bit equal to the high-bit of
      // byte 1, and the second highest bit equal to the high-bit of byte 0.
      //
      // Thus, after this block, the registers v0, v2, v4, v6 will contain 32
      // sign-bits each, in the correct order. Specifically, each byte will
      // contain two sign-bits, in the highest two bit positions.
      "sri v0.16b, v17.16b, #1\n"
      "ld2 {v16.8h, v17.8h}, [%[in_ptr]], #32\n"
      "sri v2.16b, v19.16b, #1\n"
      "sri v4.16b, v21.16b, #1\n"
      "ld2 {v18.8h, v19.8h}, [%[in_ptr]], #32\n"
      "sri v6.16b, v23.16b, #1\n"

      // The process then repeats with another sequence of unzip instructions...
      "uzp2 v8.16b, v0.16b, v2.16b\n"
      "ld2 {v20.8h, v21.8h}, [%[in_ptr]], #32\n"
      "uzp1 v30.16b, v0.16b, v2.16b\n"
      "uzp2 v10.16b, v4.16b, v6.16b\n"
      "ld2 {v22.8h, v23.8h}, [%[in_ptr]], #32\n"
      "uzp1 v31.16b, v4.16b, v6.16b\n"

      "cmp %[n], #2\n"

      // ...and SRI instructions, so that after this block, v8 and v10 will
      // contain 64 sign-bits each, in the correct order, where each byte
      // contains four sign bits in the highest four bit positions.
      "sri v8.16b, v30.16b, #2\n"
      "ld2 {v24.8h, v25.8h}, [%[in_ptr]], #32\n"
      "sri v10.16b, v31.16b, #2\n"

      // Finally, the process repeats once more so that we end up with 128
      // sign-bits in the register v0, in the correct order.
      "uzp2 v0.16b, v8.16b, v10.16b\n"
      "ld2 {v26.8h, v27.8h}, [%[in_ptr]], #32\n"
      "uzp1 v2.16b, v8.16b, v10.16b\n"
      "sri v0.16b, v2.16b, #4\n"

      "ld2 {v28.8h, v29.8h}, [%[in_ptr]], #32\n"
      "st1 {v0.2d}, [%[out_ptr]], #16\n"
      "ld2 {v30.8h, v31.8h}, [%[in_ptr]], #32\n"

      "bge 2b\n"

      "1:\n"

      // Here, n = 1, so we compute four int32 bitpacked words (from 128
      // floats) without any data reloading and then exit.

      "uzp2 v16.16b, v1.16b, v3.16b\n"
      "uzp2 v18.16b, v5.16b, v7.16b\n"
      "uzp2 v20.16b, v9.16b, v11.16b\n"
      "uzp2 v22.16b, v13.16b, v15.16b\n"
      "uzp2 v24.16b, v17.16b, v19.16b\n"
      "uzp2 v26.16b, v21.16b, v23.16b\n"
      "uzp2 v28.16b, v25.16b, v27.16b\n"
      "uzp2 v30.16b, v29.16b, v31.16b\n"

      "uzp2 v0.16b, v16.16b, v18.16b\n"
      "uzp1 v17.16b, v16.16b, v18.16b\n"
      "uzp2 v2.16b, v20.16b, v22.16b\n"
      "uzp1 v19.16b, v20.16b, v22.16b\n"
      "uzp2 v4.16b, v24.16b, v26.16b\n"
      "uzp1 v21.16b, v24.16b, v26.16b\n"
      "uzp2 v6.16b, v28.16b, v30.16b\n"
      "uzp1 v23.16b, v28.16b, v30.16b\n"
      "sri v0.16b, v17.16b, #1\n"
      "sri v2.16b, v19.16b, #1\n"
      "sri v4.16b, v21.16b, #1\n"
      "sri v6.16b, v23.16b, #1\n"

      "uzp2 v8.16b, v0.16b, v2.16b\n"
      "uzp1 v30.16b, v0.16b, v2.16b\n"
      "uzp2 v10.16b, v4.16b, v6.16b\n"
      "uzp1 v31.16b, v4.16b, v6.16b\n"
      "sri v8.16b, v30.16b, #2\n"
      "sri v10.16b, v31.16b, #2\n"

      "uzp2 v0.16b, v8.16b, v10.16b\n"
      "uzp1 v2.16b, v8.16b, v10.16b\n"
      "sri v0.16b, v2.16b, #4\n"

      "st1 {v0.2d}, [%[out_ptr]], #16\n"

      : [ in_ptr ] "+r"(input), [ n ] "+r"(num_blocks), [ out_ptr ] "+r"(output)
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31");
}

}  // namespace core
}  // namespace compute_engine

#endif
