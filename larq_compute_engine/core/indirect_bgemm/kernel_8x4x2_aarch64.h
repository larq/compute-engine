#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x2_AARCH64_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x2_AARCH64_H_

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <arm_neon.h>

#include <cstdint>
#include <type_traits>

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {
namespace kernel_8x4x2_aarch64 {

#define IF_IS_GROUPED(a) ".if %c[is_grouped]\n\t" a ".endif\n\t"

// This block of instructions takes as input the activation vector registers
// a_0, ..., a_3 and the weight vector registers w_0, ..., w_3, and computes
// 'binary multiplication and accumulation' (BMLA) using XOR and popcount
// instructions, adding the results to the 16-bit accumulator vector registers
// accum_0, ..., accum_3.
//     It also reloads data for the next loop iteration into a_0, ..., a_3 and
// w_0, ..., w_3 from the pointers a_ptr_0, ..., a_ptr_3 and w_ptr. Note that
// the accumulator loads use pairs of "load single lane" `ld1` instructions
// rather than "load and duplicate" `ld1r` instructions. This is because `ld1r`
// of 64-bit elements uses the F0/F1 (ASIMD) pipelines, whereas the 64-bit
// single-lane "ld1" instructions use only the L0/L1 (load) pipelines. See
// https://github.com/larq/compute-engine/pull/521 for more details.
#define XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64    \
  "eor v0.16b, %[a_0].16b, %[w_0].16b\n\t"  \
  "eor v1.16b, %[a_1].16b, %[w_0].16b\n\t"  \
  "eor v2.16b, %[a_2].16b, %[w_0].16b\n\t"  \
  "eor v3.16b, %[a_3].16b, %[w_0].16b\n\t"  \
  "eor v4.16b, %[a_0].16b, %[w_1].16b\n\t"  \
  "eor v5.16b, %[a_1].16b, %[w_1].16b\n\t"  \
  "eor v6.16b, %[a_2].16b, %[w_1].16b\n\t"  \
  "eor v7.16b, %[a_3].16b, %[w_1].16b\n\t"  \
  "eor v8.16b, %[a_0].16b, %[w_2].16b\n\t"  \
  "eor v9.16b, %[a_1].16b, %[w_2].16b\n\t"  \
  "eor v10.16b, %[a_2].16b, %[w_2].16b\n\t" \
  "eor v11.16b, %[a_3].16b, %[w_2].16b\n\t" \
  "ldr %q[w_0], [%[w_ptr]]\n\t"             \
  "eor v12.16b, %[a_0].16b, %[w_3].16b\n\t" \
  "eor v13.16b, %[a_1].16b, %[w_3].16b\n\t" \
  "eor v14.16b, %[a_2].16b, %[w_3].16b\n\t" \
  "eor v15.16b, %[a_3].16b, %[w_3].16b\n\t" \
  "ldr %q[w_1], [%[w_ptr], #16]\n\t"        \
  "cnt v0.16b, v0.16b\n\t"                  \
  "cnt v1.16b, v1.16b\n\t"                  \
  "ld1 {%[a_0].d}[0], [%[a_ptr_0]]\n\t"     \
  "cnt v2.16b, v2.16b\n\t"                  \
  "cnt v3.16b, v3.16b\n\t"                  \
  "ld1 {%[a_1].d}[0], [%[a_ptr_1]]\n\t"     \
  "cnt v4.16b, v4.16b\n\t"                  \
  "cnt v5.16b, v5.16b\n\t"                  \
  "ld1 {%[a_2].d}[0], [%[a_ptr_2]]\n\t"     \
  "cnt v6.16b, v6.16b\n\t"                  \
  "cnt v7.16b, v7.16b\n\t"                  \
  "ld1 {%[a_3].d}[0], [%[a_ptr_3]]\n\t"     \
  "cnt v8.16b, v8.16b\n\t"                  \
  "cnt v9.16b, v9.16b\n\t"                  \
  "ld1 {%[a_0].d}[1], [%[a_ptr_0]], #8\n\t" \
  "cnt v10.16b, v10.16b\n\t"                \
  "cnt v11.16b, v11.16b\n\t"                \
  "ld1 {%[a_1].d}[1], [%[a_ptr_1]], #8\n\t" \
  "cnt v12.16b, v12.16b\n\t"                \
  "cnt v13.16b, v13.16b\n\t"                \
  "ld1 {%[a_2].d}[1], [%[a_ptr_2]], #8\n\t" \
  "cnt v14.16b, v14.16b\n\t"                \
  "cnt v15.16b, v15.16b\n\t"                \
  "ld1 {%[a_3].d}[1], [%[a_ptr_3]], #8\n\t" \
  "addp v0.16b, v0.16b, v4.16b\n\t"         \
  "addp v1.16b, v1.16b, v5.16b\n\t"         \
  "addp v2.16b, v2.16b, v6.16b\n\t"         \
  "addp v3.16b, v3.16b, v7.16b\n\t"         \
  "ldr %q[w_2], [%[w_ptr], #32]\n\t"        \
  "addp v8.16b, v8.16b, v12.16b\n\t"        \
  "addp v9.16b, v9.16b, v13.16b\n\t"        \
  "addp v10.16b, v10.16b, v14.16b\n\t"      \
  "addp v11.16b, v11.16b, v15.16b\n\t"      \
  "ldr %q[w_3], [%[w_ptr], #48]\n\t"        \
  "addp v0.16b, v0.16b, v8.16b\n\t"         \
  "addp v1.16b, v1.16b, v9.16b\n\t"         \
  "addp v2.16b, v2.16b, v10.16b\n\t"        \
  "addp v3.16b, v3.16b, v11.16b\n\t"        \
  "add %[w_ptr], %[w_ptr], #64\n\t"         \
  "uadalp %[accum_0].8h, v0.16b\n\t"        \
  "uadalp %[accum_1].8h, v1.16b\n\t"        \
  "uadalp %[accum_2].8h, v2.16b\n\t"        \
  "uadalp %[accum_3].8h, v3.16b\n\t"

/**
 * Accumulation loops
 *
 * There are two variants of the accumulation loop: one for when we know the
 * depth is greater than one, i.e. the number of input channels is greater than
 * 64; and one for when we know the depth is equal to one, i.e. the number of
 * input channels is equal to 64. The latter case allows for a slight
 * optimisation.
 *
 * The accumulation loops use inline assembly but are equivalent to the
 * following pseudocode:
 *
 *     accum_0 = 0;
 *     accum_1 = 0;
 *     accum_2 = 0;
 *     accum_3 = 0;
 *
 *     // This block is removed in the depth=1 case
 *     // The first set of activations is already loaded, so this +1
 *     // ensures that the first 'block' loads the next set of activations.
 *     a_ptr_0 = indirection_ptr[0] + 1;
 *     a_ptr_1 = indirection_ptr[1] + 1;
 *     a_ptr_2 = indirection_ptr[2] + 1;
 *     a_ptr_3 = indirection_ptr[3] + 1;
 *     indirection_ptr += 4;
 *
 *     int fs = filter_size;
 *     do {
 *         // This loop is removed in the depth=1 case
 *         for (int d = 0; d < depth - 1; d++) {
 *             XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64
 *         }
 *
 *         // Before the final inner (depth loop) iteration, set the activation
 *         // pointers so that the activations for the next outer loop iteration
 *         // are loaded correctly.
 *         a_ptr_0 = indirection_ptr[0];
 *         a_ptr_1 = indirection_ptr[1];
 *         a_ptr_2 = indirection_ptr[2];
 *         a_ptr_3 = indirection_ptr[3];
 *         indirection_ptr += 4;
 *
 *         XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64
 *     } while (--fs > 0);
 */

template <bool IsGrouped>
inline void AccumulationLoop_Depth2OrMore(
    const std::int32_t filter_size, const std::int32_t depth,
    const std::size_t input_depth_offset, int32x4_t weights[4],
    int32x4_t activations[4], uint16x8_t accumulators[4],
    const TBitpacked*& weights_ptr, const TBitpacked* const* indirection_ptr) {
  ruy::profiler::ScopeLabel label("Accumulation loop (depth > 1)");

  // Declare these variables so we can use named registers in the ASM block.
  const TBitpacked* a_ptr_0;
  const TBitpacked* a_ptr_1;
  const TBitpacked* a_ptr_2;
  const TBitpacked* a_ptr_3;
  if (IsGrouped) {
    a_ptr_0 = indirection_ptr[0] + input_depth_offset + 2;
    a_ptr_1 = indirection_ptr[1] + input_depth_offset + 2;
    a_ptr_2 = indirection_ptr[2] + input_depth_offset + 2;
    a_ptr_3 = indirection_ptr[3] + input_depth_offset + 2;
  } else {
    a_ptr_0 = indirection_ptr[0] + 2;
    a_ptr_1 = indirection_ptr[1] + 2;
    a_ptr_2 = indirection_ptr[2] + 2;
    a_ptr_3 = indirection_ptr[3] + 2;
  }
  auto fs_index = filter_size;

  asm volatile(
      // clang-format off

      "add %[indirection_ptr], %[indirection_ptr], #32\n\t"

      // w1 is the inner (depth) loop counter
      "sub w1, %w[depth], #1\n\t"

      // Zero-out the accumulator registers
      "movi %[accum_0].8h, #0\n\t"
      "movi %[accum_1].8h, #0\n\t"
      "movi %[accum_2].8h, #0\n\t"
      "movi %[accum_3].8h, #0\n\t"

      "0:\n\t"  // Loop start label
      "subs w1, w1, #1\n\t"

      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64

      "bgt 0b\n\t"  // Inner loop branch

      "ldp %[a_ptr_0], %[a_ptr_1], [%[indirection_ptr]]\n\t"
      "ldp %[a_ptr_2], %[a_ptr_3], [%[indirection_ptr], #16]\n\t"
      "add %[indirection_ptr], %[indirection_ptr], #32\n\t"
      "sub w1, %w[depth], #1\n\t"
      "subs %w[fs_index], %w[fs_index], #1\n\t"
      IF_IS_GROUPED(
          "add %[a_ptr_0], %[a_ptr_0], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_1], %[a_ptr_1], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_2], %[a_ptr_2], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_3], %[a_ptr_3], %[input_depth_offset], lsl #2\n")
      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64

      "bgt 0b\n\t"  // Outer loop branch

      // clang-format on
      : [accum_0] "=&w"(accumulators[0]), [accum_1] "=&w"(accumulators[1]),
        [accum_2] "=&w"(accumulators[2]), [accum_3] "=&w"(accumulators[3]),
        [w_ptr] "+r"(weights_ptr), [indirection_ptr] "+r"(indirection_ptr),
        [a_ptr_0] "+r"(a_ptr_0), [a_ptr_1] "+r"(a_ptr_1),
        [a_ptr_2] "+r"(a_ptr_2), [a_ptr_3] "+r"(a_ptr_3),
        [fs_index] "+r"(fs_index)
      : [w_0] "w"(weights[0]), [w_1] "w"(weights[1]), [w_2] "w"(weights[2]),
        [w_3] "w"(weights[3]), [a_0] "w"(activations[0]),
        [a_1] "w"(activations[1]), [a_2] "w"(activations[2]),
        [a_3] "w"(activations[3]), [depth] "r"(depth),
        [input_depth_offset] "r"(input_depth_offset),
        [is_grouped] "i"(IsGrouped)
      : "cc", "memory", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
}

template <bool IsGrouped>
inline void AccumulationLoop_Depth1(
    const std::int32_t filter_size, const std::size_t input_depth_offset,
    int32x4_t weights[4], int32x4_t activations[4], uint16x8_t accumulators[4],
    const TBitpacked*& weights_ptr, const TBitpacked* const* indirection_ptr) {
  ruy::profiler::ScopeLabel label("Accumulation loop (depth = 1)");

  // Declare these variables so we can use named registers in the ASM block.
  const TBitpacked* a_ptr_0;
  const TBitpacked* a_ptr_1;
  const TBitpacked* a_ptr_2;
  const TBitpacked* a_ptr_3;
  auto fs_index = filter_size;

  asm volatile(
      // clang-format off

      "add %[indirection_ptr], %[indirection_ptr], #32\n\t"

      // Zero-out the accumulator registers
      "movi %[accum_0].8h, #0\n\t"
      "movi %[accum_1].8h, #0\n\t"
      "movi %[accum_2].8h, #0\n\t"
      "movi %[accum_3].8h, #0\n\t"

      "0:\n\t"  // Loop label

      "ldp %[a_ptr_0], %[a_ptr_1], [%[indirection_ptr]]\n\t"
      "ldp %[a_ptr_2], %[a_ptr_3], [%[indirection_ptr], #16]\n\t"
      "add %[indirection_ptr], %[indirection_ptr], #32\n\t"
      "subs %w[fs_index], %w[fs_index], #1\n\t"
      IF_IS_GROUPED(
          "add %[a_ptr_0], %[a_ptr_0], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_1], %[a_ptr_1], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_2], %[a_ptr_2], %[input_depth_offset], lsl #2\n"
          "add %[a_ptr_3], %[a_ptr_3], %[input_depth_offset], lsl #2\n")
      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64

      "bgt 0b\n\t"  // Loop branch

      // clang-format on
      : [accum_0] "=&w"(accumulators[0]), [accum_1] "=&w"(accumulators[1]),
        [accum_2] "=&w"(accumulators[2]), [accum_3] "=&w"(accumulators[3]),
        [w_ptr] "+r"(weights_ptr), [indirection_ptr] "+r"(indirection_ptr),
        [a_ptr_0] "=&r"(a_ptr_0), [a_ptr_1] "=&r"(a_ptr_1),
        [a_ptr_2] "=&r"(a_ptr_2), [a_ptr_3] "=&r"(a_ptr_3),
        [fs_index] "+r"(fs_index)
      : [w_0] "w"(weights[0]), [w_1] "w"(weights[1]), [w_2] "w"(weights[2]),
        [w_3] "w"(weights[3]), [a_0] "w"(activations[0]),
        [a_1] "w"(activations[1]), [a_2] "w"(activations[2]),
        [a_3] "w"(activations[3]), [input_depth_offset] "r"(input_depth_offset),
        [is_grouped] "i"(IsGrouped)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15");
}

#undef XOR_POPCOUNT_ACCUM_LOAD_BLOCK_64

/**
 * Output transforms
 *
 * These destination-type-specific functions take the accumulator values and
 * perform the output transform, before storing the results in the output array.
 * They additionally reload data into the weight and activation registers for
 * the first iteration of the next accumulation loop.
 */

// Float output transform
template <bool IsGrouped>
inline void OutputTransformAndLoadNextAndStore(
    std::int32_t& c_out_index, const std::size_t input_depth_offset,
    const std::int32_t group_end_output_channel, uint16x8_t accumulators[4],
    int32x4_t weights[4], int32x4_t activations[4],
    const bconv2d::OutputTransform<float>& output_transform,
    const TBitpacked* weights_ptr, const TBitpacked* const* indirection_ptr,
    float*& output_ptr_0, float*& output_ptr_1, float*& output_ptr_2,
    float*& output_ptr_3) {
  ruy::profiler::ScopeLabel label("Float output transform and store");

  // Declare result registers.
  float32x4x2_t results[4];

  asm("ldp x0, x1, [%[indirection_ptr]]\n\t"
      "ldp x2, x3, [%[indirection_ptr], #16]\n\t"

      // Use "unsigned shift left long" instructions to perform the
      // back-transformation left-shift and extend the result to int32.
      "ld1r {v0.4s}, [%[clamp_min_addr]]\n\t"
      "ushll %[result_00].4s, %[accum_0].4h, #1\n\t"
      "ushll %[result_10].4s, %[accum_1].4h, #1\n\t"
      "ld1r {v1.4s}, [%[clamp_max_addr]]\n\t"
      "ushll %[result_20].4s, %[accum_2].4h, #1\n\t"
      "ushll %[result_30].4s, %[accum_3].4h, #1\n\t"
      "ldr q2, [%[multiplier_addr]]\n\t"
      "ushll2 %[result_31].4s, %[accum_3].8h, #1\n\t"
      "ushll2 %[result_21].4s, %[accum_2].8h, #1\n\t"
      "ldr q3, [%[multiplier_addr], #16]\n\t"
      "ushll2 %[result_11].4s, %[accum_1].8h, #1\n\t"
      "ushll2 %[result_01].4s, %[accum_0].8h, #1\n\t"

      IF_IS_GROUPED("add x0, x0, %[input_depth_offset], lsl #2\n"
                    "add x1, x1, %[input_depth_offset], lsl #2\n"
                    "add x2, x2, %[input_depth_offset], lsl #2\n"
                    "add x3, x3, %[input_depth_offset], lsl #2\n")

      // Perform clamping
      "ldr q4, [%[bias_addr]]\n\t"
      "smax %[result_30].4s, %[result_30].4s, v0.4s\n\t"
      "smax %[result_31].4s, %[result_31].4s, v0.4s\n\t"
      "ldr q5, [%[bias_addr], #16]\n\t"
      "smax %[result_20].4s, %[result_20].4s, v0.4s\n\t"
      "smax %[result_21].4s, %[result_21].4s, v0.4s\n\t"
      "ldr %q[w_0], [%[w_ptr], #-64]\n\t"
      "smax %[result_10].4s, %[result_10].4s, v0.4s\n\t"
      "smax %[result_11].4s, %[result_11].4s, v0.4s\n\t"
      "ldr %q[w_1], [%[w_ptr], #-48]\n\t"
      "smax %[result_00].4s, %[result_00].4s, v0.4s\n\t"
      "smax %[result_01].4s, %[result_01].4s, v0.4s\n\t"
      "ld1 {%[a_0].d}[0], [x0]\n\t"
      "smin %[result_30].4s, %[result_30].4s, v1.4s\n\t"
      "smin %[result_31].4s, %[result_31].4s, v1.4s\n\t"
      "ld1 {%[a_1].d}[0], [x1]\n\t"
      "smin %[result_20].4s, %[result_20].4s, v1.4s\n\t"
      "smin %[result_21].4s, %[result_21].4s, v1.4s\n\t"
      "ld1 {%[a_2].d}[0], [x2]\n\t"
      "smin %[result_10].4s, %[result_10].4s, v1.4s\n\t"
      "smin %[result_11].4s, %[result_11].4s, v1.4s\n\t"
      "ld1 {%[a_3].d}[0], [x3]\n\t"
      "smin %[result_00].4s, %[result_00].4s, v1.4s\n\t"
      "smin %[result_01].4s, %[result_01].4s, v1.4s\n\t"

      // Convert to float, multiply by the multiplier, and add the bias. Note
      // that the float conversion instructions ("scvtf") are *very* slow and
      // block the Neon pipeline. It is therefore important for optimal
      // performance to interleave the float multiply and add instructions
      // between the "scvtf" instructions.
      "ld1 {%[a_0].d}[1], [x0]\n\t"
      "scvtf %[result_30].4s, %[result_30].4s\n\t"
      "scvtf %[result_31].4s, %[result_31].4s\n\t"
      "ld1 {%[a_1].d}[1], [x1]\n\t"
      "scvtf %[result_20].4s, %[result_20].4s\n\t"
      "fmul %[result_30].4s, %[result_30].4s, v2.4s\n\t"
      "scvtf %[result_21].4s, %[result_21].4s\n\t"
      "ld1 {%[a_2].d}[1], [x2]\n\t"
      "fmul %[result_31].4s, %[result_31].4s, v3.4s\n\t"
      "fadd %[result_30].4s, %[result_30].4s, v4.4s\n\t"
      "scvtf %[result_10].4s, %[result_10].4s\n\t"
      "ld1 {%[a_3].d}[1], [x3]\n\t"
      "fmul %[result_20].4s, %[result_20].4s, v2.4s\n\t"
      "fadd %[result_31].4s, %[result_31].4s, v5.4s\n\t"
      "scvtf %[result_11].4s, %[result_11].4s\n\t"
      "ldr %q[w_2], [%[w_ptr], #-32]\n\t"
      "fmul %[result_21].4s, %[result_21].4s, v3.4s\n\t"
      "fadd %[result_20].4s, %[result_20].4s, v4.4s\n\t"
      "scvtf %[result_00].4s, %[result_00].4s\n\t"
      "ldr %q[w_3], [%[w_ptr], #-16]\n\t"
      "fmul %[result_10].4s, %[result_10].4s, v2.4s\n\t"
      "fadd %[result_21].4s, %[result_21].4s, v5.4s\n\t"
      "scvtf %[result_01].4s, %[result_01].4s\n\t"
      "fmul %[result_11].4s, %[result_11].4s, v3.4s\n\t"
      "fadd %[result_10].4s, %[result_10].4s, v4.4s\n\t"
      "fmul %[result_00].4s, %[result_00].4s, v2.4s\n\t"
      "fmul %[result_01].4s, %[result_01].4s, v3.4s\n\t"
      "fadd %[result_11].4s, %[result_11].4s, v5.4s\n\t"
      "fadd %[result_00].4s, %[result_00].4s, v4.4s\n\t"
      "fadd %[result_01].4s, %[result_01].4s, v5.4s\n\t"
      :
      [w_0] "=&w"(weights[0]), [w_1] "=&w"(weights[1]), [w_2] "=&w"(weights[2]),
      [w_3] "=&w"(weights[3]), [a_0] "=&w"(activations[0]),
      [a_1] "=&w"(activations[1]), [a_2] "=&w"(activations[2]),
      [a_3] "=&w"(activations[3]), [result_00] "=&w"(results[0].val[0]),
      [result_01] "=&w"(results[0].val[1]),
      [result_10] "=&w"(results[1].val[0]),
      [result_11] "=&w"(results[1].val[1]),
      [result_20] "=&w"(results[2].val[0]),
      [result_21] "=&w"(results[2].val[1]),
      [result_30] "=&w"(results[3].val[0]), [result_31] "=&w"(results[3].val[1])
      : [accum_0] "w"(accumulators[0]), [accum_1] "w"(accumulators[1]),
        [accum_2] "w"(accumulators[2]), [accum_3] "w"(accumulators[3]),
        [clamp_min_addr] "r"(&output_transform.clamp_min),
        [clamp_max_addr] "r"(&output_transform.clamp_max),
        [multiplier_addr] "r"(output_transform.multiplier + c_out_index),
        [bias_addr] "r"(output_transform.bias + c_out_index),
        [w_ptr] "r"(weights_ptr), [indirection_ptr] "r"(indirection_ptr),
        [input_depth_offset] "r"(input_depth_offset),
        [is_grouped] "i"(IsGrouped)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5");

  if (LCE_LIKELY(group_end_output_channel - c_out_index >= 8)) {
    vst1q_f32(output_ptr_3, results[3].val[0]);
    vst1q_f32(output_ptr_3 + 4, results[3].val[1]);
    output_ptr_3 += 8;
    vst1q_f32(output_ptr_2, results[2].val[0]);
    vst1q_f32(output_ptr_2 + 4, results[2].val[1]);
    output_ptr_2 += 8;
    vst1q_f32(output_ptr_1, results[1].val[0]);
    vst1q_f32(output_ptr_1 + 4, results[1].val[1]);
    output_ptr_1 += 8;
    vst1q_f32(output_ptr_0, results[0].val[0]);
    vst1q_f32(output_ptr_0 + 4, results[0].val[1]);
    output_ptr_0 += 8;

    c_out_index += 8;
  } else {
#define STORE_SINGLE_ELEMENT_LOW(index)                     \
  vst1q_lane_f32(output_ptr_3++, results[3].val[0], index); \
  vst1q_lane_f32(output_ptr_2++, results[2].val[0], index); \
  vst1q_lane_f32(output_ptr_1++, results[1].val[0], index); \
  vst1q_lane_f32(output_ptr_0++, results[0].val[0], index);
#define STORE_SINGLE_ELEMENT_HIGH(index)                    \
  vst1q_lane_f32(output_ptr_3++, results[3].val[1], index); \
  vst1q_lane_f32(output_ptr_2++, results[2].val[1], index); \
  vst1q_lane_f32(output_ptr_1++, results[1].val[1], index); \
  vst1q_lane_f32(output_ptr_0++, results[0].val[1], index);
    STORE_SINGLE_ELEMENT_LOW(0)
    if (group_end_output_channel - c_out_index >= 2) {
      STORE_SINGLE_ELEMENT_LOW(1)
      if (group_end_output_channel - c_out_index >= 3) {
        STORE_SINGLE_ELEMENT_LOW(2)
        if (group_end_output_channel - c_out_index >= 4) {
          STORE_SINGLE_ELEMENT_LOW(3)
          if (group_end_output_channel - c_out_index >= 5) {
            STORE_SINGLE_ELEMENT_HIGH(0)
            if (group_end_output_channel - c_out_index >= 6) {
              STORE_SINGLE_ELEMENT_HIGH(1)
              if (group_end_output_channel - c_out_index >= 7) {
                STORE_SINGLE_ELEMENT_HIGH(2)
              }
            }
          }
        }
      }
    }
#undef STORE_SINGLE_ELEMENT_LOW
#undef STORE_SINGLE_ELEMENT_HIGH

    c_out_index = group_end_output_channel;
  }
}

// Int8 output transform
template <bool IsGrouped>
inline void OutputTransformAndLoadNextAndStore(
    std::int32_t& c_out_index, const std::size_t input_depth_offset,
    const std::int32_t group_end_output_channel, uint16x8_t accumulators[4],
    int32x4_t weights[4], int32x4_t activations[4],
    const bconv2d::OutputTransform<std::int8_t>& output_transform,
    const TBitpacked* weights_ptr, const TBitpacked* const* indirection_ptr,
    std::int8_t*& output_ptr_0, std::int8_t*& output_ptr_1,
    std::int8_t*& output_ptr_2, std::int8_t*& output_ptr_3) {
  ruy::profiler::ScopeLabel label("Int8 output transform and store");

  // Declare result registers. These are wider than we need for just the final
  // int8 values, which is necessary for intermediate results.
  int8x16x2_t results[4];

  asm("ldp x0, x1, [%[indirection_ptr]]\n\t"
      "ldp x2, x3, [%[indirection_ptr], #16]\n\t"

      // Use "unsigned shift left long" instructions to perform the
      // back-transformation left-shift and extend the result to int32.
      "ld1r {v0.4s}, [%[clamp_min_addr]]\n\t"
      "ushll %[result_00].4s, %[accum_0].4h, #1\n\t"
      "ushll %[result_10].4s, %[accum_1].4h, #1\n\t"
      "ld1r {v1.4s}, [%[clamp_max_addr]]\n\t"
      "ushll %[result_20].4s, %[accum_2].4h, #1\n\t"
      "ushll %[result_30].4s, %[accum_3].4h, #1\n\t"
      "ldr q2, [%[multiplier_addr]]\n\t"
      "ushll2 %[result_31].4s, %[accum_3].8h, #1\n\t"
      "ushll2 %[result_21].4s, %[accum_2].8h, #1\n\t"
      "ldr q3, [%[multiplier_addr], #16]\n\t"
      "ushll2 %[result_11].4s, %[accum_1].8h, #1\n\t"
      "ushll2 %[result_01].4s, %[accum_0].8h, #1\n\t"

      IF_IS_GROUPED("add x0, x0, %[input_depth_offset], lsl #2\n"
                    "add x1, x1, %[input_depth_offset], lsl #2\n"
                    "add x2, x2, %[input_depth_offset], lsl #2\n"
                    "add x3, x3, %[input_depth_offset], lsl #2\n")

      // Perform clamping
      "ldr q4, [%[bias_addr]]\n\t"
      "smax %[result_30].4s, %[result_30].4s, v0.4s\n\t"
      "smax %[result_31].4s, %[result_31].4s, v0.4s\n\t"
      "ldr q5, [%[bias_addr], #16]\n\t"
      "smax %[result_20].4s, %[result_20].4s, v0.4s\n\t"
      "smax %[result_21].4s, %[result_21].4s, v0.4s\n\t"
      "ldr %q[w_0], [%[w_ptr], #-64]\n\t"
      "smax %[result_10].4s, %[result_10].4s, v0.4s\n\t"
      "smax %[result_11].4s, %[result_11].4s, v0.4s\n\t"
      "ldr %q[w_1], [%[w_ptr], #-48]\n\t"
      "smax %[result_00].4s, %[result_00].4s, v0.4s\n\t"
      "smax %[result_01].4s, %[result_01].4s, v0.4s\n\t"
      "ld1 {%[a_0].d}[0], [x0]\n\t"
      "smin %[result_30].4s, %[result_30].4s, v1.4s\n\t"
      "smin %[result_31].4s, %[result_31].4s, v1.4s\n\t"
      "ld1 {%[a_1].d}[0], [x1]\n\t"
      "smin %[result_20].4s, %[result_20].4s, v1.4s\n\t"
      "smin %[result_21].4s, %[result_21].4s, v1.4s\n\t"
      "ld1 {%[a_2].d}[0], [x2]\n\t"
      "smin %[result_10].4s, %[result_10].4s, v1.4s\n\t"
      "smin %[result_11].4s, %[result_11].4s, v1.4s\n\t"
      "ld1 {%[a_3].d}[0], [x3]\n\t"
      "smin %[result_00].4s, %[result_00].4s, v1.4s\n\t"
      "smin %[result_01].4s, %[result_01].4s, v1.4s\n\t"

      // Convert to float, multiply by the multiplier, add the bias, convert
      // back to integer, and use a series of "saturating extract narrow"
      // instructions to narrow the result to Int8. Note that the float
      // conversion instructions ("scvtf" and "fcvtns") are *very* slow and
      // block the Neon pipeline. It is therefore important for optimal
      // performance to interleave other instructions between them.
      "ld1 {%[a_0].d}[1], [x0]\n\t"
      "scvtf %[result_30].4s, %[result_30].4s\n\t"
      "scvtf %[result_31].4s, %[result_31].4s\n\t"
      "ld1 {%[a_1].d}[1], [x1]\n\t"
      "scvtf %[result_20].4s, %[result_20].4s\n\t"
      "fmul %[result_30].4s, %[result_30].4s, v2.4s\n\t"
      "scvtf %[result_21].4s, %[result_21].4s\n\t"
      "ld1 {%[a_2].d}[1], [x2]\n\t"
      "fmul %[result_31].4s, %[result_31].4s, v3.4s\n\t"
      "fadd %[result_30].4s, %[result_30].4s, v4.4s\n\t"
      "scvtf %[result_10].4s, %[result_10].4s\n\t"
      "ld1 {%[a_3].d}[1], [x3]\n\t"
      "fmul %[result_20].4s, %[result_20].4s, v2.4s\n\t"
      "fadd %[result_31].4s, %[result_31].4s, v5.4s\n\t"
      "scvtf %[result_11].4s, %[result_11].4s\n\t"
      "ldr %q[w_2], [%[w_ptr], #-32]\n\t"
      "fmul %[result_21].4s, %[result_21].4s, v3.4s\n\t"
      "fadd %[result_20].4s, %[result_20].4s, v4.4s\n\t"
      "fcvtns %[result_30].4s, %[result_30].4s\n\t"
      "ldr %q[w_3], [%[w_ptr], #-16]\n\t"
      "fmul %[result_10].4s, %[result_10].4s, v2.4s\n\t"
      "fadd %[result_21].4s, %[result_21].4s, v5.4s\n\t"
      "fcvtns %[result_31].4s, %[result_31].4s\n\t"
      "sqxtn %[result_30].4h, %[result_30].4s\n\t"
      "fmul %[result_11].4s, %[result_11].4s, v3.4s\n\t"
      "scvtf %[result_00].4s, %[result_00].4s\n\t"
      "sqxtn2 %[result_30].8h, %[result_31].4s\n\t"
      "fadd %[result_10].4s, %[result_10].4s, v4.4s\n\t"
      "scvtf %[result_01].4s, %[result_01].4s\n\t"
      "fmul %[result_00].4s, %[result_00].4s, v2.4s\n\t"
      "fadd %[result_11].4s, %[result_11].4s, v5.4s\n\t"
      "fcvtns %[result_20].4s, %[result_20].4s\n\t"
      "fmul %[result_01].4s, %[result_01].4s, v3.4s\n\t"
      "fadd %[result_00].4s, %[result_00].4s, v4.4s\n\t"
      "fcvtns %[result_21].4s, %[result_21].4s\n\t"
      "sqxtn %[result_20].4h, %[result_20].4s\n\t"
      "fadd %[result_01].4s, %[result_01].4s, v5.4s\n\t"
      "fcvtns %[result_10].4s, %[result_10].4s\n\t"
      "sqxtn2 %[result_20].8h, %[result_21].4s\n\t"
      "fcvtns %[result_11].4s, %[result_11].4s\n\t"
      "sqxtn %[result_10].4h, %[result_10].4s\n\t"
      "sqxtn %[result_30].8b, %[result_30].8h\n\t"
      "fcvtns %[result_00].4s, %[result_00].4s\n\t"
      "sqxtn2 %[result_10].8h, %[result_11].4s\n\t"
      "sqxtn %[result_20].8b, %[result_20].8h\n\t"
      "fcvtns %[result_01].4s, %[result_01].4s\n\t"
      "sqxtn %[result_00].4h, %[result_00].4s\n\t"
      "sqxtn2 %[result_00].8h, %[result_01].4s\n\t"
      "sqxtn %[result_10].8b, %[result_10].8h\n\t"
      "sqxtn %[result_00].8b, %[result_00].8h\n\t"
      :
      [w_0] "=&w"(weights[0]), [w_1] "=&w"(weights[1]), [w_2] "=&w"(weights[2]),
      [w_3] "=&w"(weights[3]), [a_0] "=&w"(activations[0]),
      [a_1] "=&w"(activations[1]), [a_2] "=&w"(activations[2]),
      [a_3] "=&w"(activations[3]), [result_00] "=&w"(results[0].val[0]),
      [result_01] "=&w"(results[0].val[1]),
      [result_10] "=&w"(results[1].val[0]),
      [result_11] "=&w"(results[1].val[1]),
      [result_20] "=&w"(results[2].val[0]),
      [result_21] "=&w"(results[2].val[1]),
      [result_30] "=&w"(results[3].val[0]), [result_31] "=&w"(results[3].val[1])
      : [accum_0] "w"(accumulators[0]), [accum_1] "w"(accumulators[1]),
        [accum_2] "w"(accumulators[2]), [accum_3] "w"(accumulators[3]),
        [clamp_min_addr] "r"(&output_transform.clamp_min),
        [clamp_max_addr] "r"(&output_transform.clamp_max),
        [multiplier_addr] "r"(output_transform.multiplier + c_out_index),
        [bias_addr] "r"(output_transform.bias + c_out_index),
        [w_ptr] "r"(weights_ptr), [indirection_ptr] "r"(indirection_ptr),
        [input_depth_offset] "r"(input_depth_offset),
        [is_grouped] "i"(IsGrouped)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5");

  if (LCE_LIKELY(group_end_output_channel - c_out_index >= 8)) {
    vst1_s8(output_ptr_3, vget_low_s8(results[3].val[0]));
    output_ptr_3 += 8;
    vst1_s8(output_ptr_2, vget_low_s8(results[2].val[0]));
    output_ptr_2 += 8;
    vst1_s8(output_ptr_1, vget_low_s8(results[1].val[0]));
    output_ptr_1 += 8;
    vst1_s8(output_ptr_0, vget_low_s8(results[0].val[0]));
    output_ptr_0 += 8;

    c_out_index += 8;
  } else {
#define STORE_SINGLE_ELEMENT(index)                                    \
  vst1_lane_s8(output_ptr_3++, vget_low_s8(results[3].val[0]), index); \
  vst1_lane_s8(output_ptr_2++, vget_low_s8(results[2].val[0]), index); \
  vst1_lane_s8(output_ptr_1++, vget_low_s8(results[1].val[0]), index); \
  vst1_lane_s8(output_ptr_0++, vget_low_s8(results[0].val[0]), index);
    STORE_SINGLE_ELEMENT(0)
    if (group_end_output_channel - c_out_index >= 2) {
      STORE_SINGLE_ELEMENT(1)
      if (group_end_output_channel - c_out_index >= 3) {
        STORE_SINGLE_ELEMENT(2)
        if (group_end_output_channel - c_out_index >= 4) {
          STORE_SINGLE_ELEMENT(3)
          if (group_end_output_channel - c_out_index >= 5) {
            STORE_SINGLE_ELEMENT(4)
            if (group_end_output_channel - c_out_index >= 6) {
              STORE_SINGLE_ELEMENT(5)
              if (group_end_output_channel - c_out_index >= 7) {
                STORE_SINGLE_ELEMENT(6)
              }
            }
          }
        }
      }
    }
#undef STORE_SINGLE_ELEMENT

    c_out_index = group_end_output_channel;
  }
}

#undef IF_IS_GROUPED

}  // namespace kernel_8x4x2_aarch64

/**
 * A 8x4x2 Neon micro-kernel for float or int8 output on Aarch64.
 */
template <typename DstScalar, bool Depth2OrMore, bool IsGrouped>
class Kernel8x4x2Aarch64 : public Kernel {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");
  static_assert(sizeof(TBitpacked) == 4, "");

  const bconv2d::OutputTransform<DstScalar> output_transform;

 public:
  Kernel8x4x2Aarch64(
      const bconv2d::BConv2DParams* bconv2d_params,
      const RuntimeShape& bitpacked_input_shape,
      const RuntimeShape& output_shape,
      const bconv2d::OutputTransform<DstScalar>& output_transform)
      : Kernel(8, 4, 2, bconv2d_params, bitpacked_input_shape, output_shape),
        output_transform(output_transform) {}

  void Run(const std::int32_t pixel_start, const std::int32_t pixel_end,
           void* output_ptr) const override {
    ruy::profiler::ScopeLabel label(
        "Indirect BGEMM block (8x4x2, Aarch64 optimised)");

    TFLITE_DCHECK_GE(this->input_depth, 1);
    TFLITE_DCHECK_GE(this->output_channels, 1);
    TFLITE_DCHECK_GE(this->filter_size, 1);
    TFLITE_DCHECK_GE(this->groups, 1);
    TFLITE_DCHECK_EQ(this->input_depth % this->groups, 0);
    TFLITE_DCHECK_EQ(this->output_channels % this->groups, 0);

    const auto input_depth_per_group = this->input_depth / this->groups;
    const auto output_channels_per_group = this->output_channels / this->groups;

    if (this->filter_size < 1 || input_depth_per_group < 1 ||
        output_channels_per_group < 1) {
      return;
    }

    for (std::int32_t p_index = pixel_start; p_index < pixel_end;
         p_index += 4) {
      const TBitpacked* weights_ptr = this->packed_weights.data();
      const TBitpacked* const* indirection_ptr =
          this->indirection_buffer.data() + p_index * this->filter_size;
      auto output_ptr_0 = reinterpret_cast<DstScalar*>(output_ptr) +
                          p_index * this->output_channels;
      auto output_ptr_1 = output_ptr_0 + this->output_channels;
      auto output_ptr_2 = output_ptr_1 + this->output_channels;
      auto output_ptr_3 = output_ptr_2 + this->output_channels;

      // At the end of the output array we might get a block where the number of
      // pixels is less than 4. When this happens we set the 'leftover' output
      // pointer equal to the first output pointer, so that there's no risk of
      // writing beyond the array bounds. At the end we write to the output
      // array 'back to front' so that the outputs for the first pixel are
      // written last, which means that the result will still be correct.
      const std::int32_t remaining_pixels = pixel_end - p_index;
      if (remaining_pixels < 4) {
        output_ptr_3 = output_ptr_0;
        if (remaining_pixels < 3) {
          output_ptr_2 = output_ptr_0;
          if (remaining_pixels < 2) {
            output_ptr_1 = output_ptr_0;
          }
        }
      }

      // Pre-load weights and activations.
      int32x4_t weights[4] = {
          vld1q_s32(weights_ptr), vld1q_s32(weights_ptr + 4),
          vld1q_s32(weights_ptr + 8), vld1q_s32(weights_ptr + 12)};
      weights_ptr += 16;
      int32x4_t activations[4] = {
          vreinterpretq_s32_s64(
              vld1q_dup_s64((std::int64_t*)indirection_ptr[0])),
          vreinterpretq_s32_s64(
              vld1q_dup_s64((std::int64_t*)indirection_ptr[1])),
          vreinterpretq_s32_s64(
              vld1q_dup_s64((std::int64_t*)indirection_ptr[2])),
          vreinterpretq_s32_s64(
              vld1q_dup_s64((std::int64_t*)indirection_ptr[3]))};

      std::size_t input_depth_offset = 0;
      std::int32_t group_end_output_channel = output_channels_per_group;

      std::int32_t c_out_index = 0;
      do {
        uint16x8_t accumulators[4];

        if (Depth2OrMore) {
          kernel_8x4x2_aarch64::AccumulationLoop_Depth2OrMore<IsGrouped>(
              this->filter_size, input_depth_per_group / 2, input_depth_offset,
              weights, activations, accumulators, weights_ptr, indirection_ptr);
        } else {
          kernel_8x4x2_aarch64::AccumulationLoop_Depth1<IsGrouped>(
              this->filter_size, input_depth_offset, weights, activations,
              accumulators, weights_ptr, indirection_ptr);
        }

        std::int32_t next_group_end_output_channel = group_end_output_channel;
        if (IsGrouped && c_out_index >= group_end_output_channel - 8) {
          input_depth_offset += input_depth_per_group;
          next_group_end_output_channel =
              group_end_output_channel + output_channels_per_group;
        }

        kernel_8x4x2_aarch64::OutputTransformAndLoadNextAndStore<IsGrouped>(
            c_out_index, input_depth_offset, group_end_output_channel,
            accumulators, weights, activations, output_transform, weights_ptr,
            indirection_ptr, output_ptr_0, output_ptr_1, output_ptr_2,
            output_ptr_3);

        if (IsGrouped) {
          group_end_output_channel = next_group_end_output_channel;
        }
      } while (c_out_index < this->output_channels);
    }
  }
};

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x2_AARCH64_H_
