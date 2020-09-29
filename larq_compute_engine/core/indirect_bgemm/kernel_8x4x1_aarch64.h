#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x1_AARCH64_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x1_AARCH64_H_

#ifndef __aarch64__
#pragma GCC error "ERROR: This file should only be compiled for Aarch64."
#endif

#include <arm_neon.h>

#include <cstdint>
#include <type_traits>

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {
namespace kernel_8x4x1_aarch64 {

/**
 * This block of instructions takes as input the activation vector registers
 * a_0, ..., a_3 and the weight vector registers w_0, w_1, and computes 'binary
 * multiplication and accumulation' (BMLA) using XOR and popcount instructions,
 * adding the results to the 16-bit accumulator vector registers accum_0, ...,
 * accum_3.
 *
 * Additionally, it loads the next values into a_0, ..., a_3 (from the pointers
 * a_ptr_0, ..., a_ptr_3) and into w_0, w_1 (from the pointer w_ptr).
 */
#define XOR_POPCOUNT_ACCUM_LOAD_BLOCK_32   \
  "eor v0.16b, %[a_0].16b, %[w_0].16b\n\t" \
  "eor v1.16b, %[a_1].16b, %[w_0].16b\n\t" \
  "eor v2.16b, %[a_2].16b, %[w_0].16b\n\t" \
  "eor v3.16b, %[a_3].16b, %[w_0].16b\n\t" \
  "eor v4.16b, %[a_0].16b, %[w_1].16b\n\t" \
  "eor v5.16b, %[a_1].16b, %[w_1].16b\n\t" \
  "ldr %q[w_0], [%[w_ptr]]\n\t"            \
  "eor v6.16b, %[a_2].16b, %[w_1].16b\n\t" \
  "eor v7.16b, %[a_3].16b, %[w_1].16b\n\t" \
  "ldr %q[w_1], [%[w_ptr], #16]\n\t"       \
  "cnt v0.16b, v0.16b\n\t"                 \
  "cnt v1.16b, v1.16b\n\t"                 \
  "ld1r {%[a_0].4s}, [%[a_ptr_0]], #4\n\t" \
  "cnt v2.16b, v2.16b\n\t"                 \
  "cnt v3.16b, v3.16b\n\t"                 \
  "ld1r {%[a_1].4s}, [%[a_ptr_1]], #4\n\t" \
  "cnt v4.16b, v4.16b\n\t"                 \
  "cnt v5.16b, v5.16b\n\t"                 \
  "ld1r {%[a_2].4s}, [%[a_ptr_2]], #4\n\t" \
  "cnt v6.16b, v6.16b\n\t"                 \
  "cnt v7.16b, v7.16b\n\t"                 \
  "ld1r {%[a_3].4s}, [%[a_ptr_3]], #4\n\t" \
  "addp v0.16b, v0.16b, v4.16b\n\t"        \
  "addp v1.16b, v1.16b, v5.16b\n\t"        \
  "addp v2.16b, v2.16b, v6.16b\n\t"        \
  "addp v3.16b, v3.16b, v7.16b\n\t"        \
  "add %[w_ptr], %[w_ptr], #32\n\t"        \
  "uadalp %[accum_0].8h, v0.16b\n\t"       \
  "uadalp %[accum_1].8h, v1.16b\n\t"       \
  "uadalp %[accum_2].8h, v2.16b\n\t"       \
  "uadalp %[accum_3].8h, v3.16b\n\t"

/**
 * This function performs the accumulation loop when we know that the depth is
 * greater than one, i.e. the (bitpacked) number of input channels is greater
 * than 32.
 */
inline void AccumulationLoop_Depth2OrMore(
    std::int32_t conv_kernel_size, const std::int32_t depth,
    int32x4_t weights[2], int32x4_t activations[4], uint16x8_t accumulators[4],
    TBitpacked*& weights_ptr, TBitpacked** indirection_buffer) {
  ruy::profiler::ScopeLabel label("Accumulation loop (depth > 1)");

  // Declare these variables so we can use named registers in the ASM block.
  TBitpacked** indirection_buffer_ = indirection_buffer + 4;
  TBitpacked* a_ptr_0 = indirection_buffer[0] + 1;
  TBitpacked* a_ptr_1 = indirection_buffer[1] + 1;
  TBitpacked* a_ptr_2 = indirection_buffer[2] + 1;
  TBitpacked* a_ptr_3 = indirection_buffer[3] + 1;

  asm volatile(
      // w0 is the outer (kernel size) loop counter
      "mov w0, %w[kernel_size]\n\t"

      // w1 is the inner (depth) loop counter
      "sub w1, %w[depth], #1\n\t"

      // Zero-out the accumulator registers
      "movi %[accum_0].8h, #0\n\t"
      "movi %[accum_1].8h, #0\n\t"
      "movi %[accum_2].8h, #0\n\t"
      "movi %[accum_3].8h, #0\n\t"

      "0:\n\t"  // Loop start label
      "subs w1, w1, #1\n\t"

      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_32

      "bgt 0b\n\t"  // Inner loop branch

      "ldp %[a_ptr_0], %[a_ptr_1], [%[indirection_buffer]]\n\t"
      "ldp %[a_ptr_2], %[a_ptr_3], [%[indirection_buffer], #16]\n\t"
      "add %[indirection_buffer], %[indirection_buffer], #32\n\t"
      "sub w1, %w[depth], #1\n\t"
      "subs w0, w0, #1\n\t"

      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_32

      "bgt 0b\n\t"  // Outer loop branch
      : [ accum_0 ] "=&w"(accumulators[0]), [ accum_1 ] "=&w"(accumulators[1]),
        [ accum_2 ] "=&w"(accumulators[2]), [ accum_3 ] "=&w"(accumulators[3]),
        [ w_ptr ] "+r"(weights_ptr),
        [ indirection_buffer ] "+r"(indirection_buffer_),
        [ a_ptr_0 ] "+r"(a_ptr_0), [ a_ptr_1 ] "+r"(a_ptr_1),
        [ a_ptr_2 ] "+r"(a_ptr_2), [ a_ptr_3 ] "+r"(a_ptr_3)
      : [ w_0 ] "w"(weights[0]), [ w_1 ] "w"(weights[1]),
        [ a_0 ] "w"(activations[0]), [ a_1 ] "w"(activations[1]),
        [ a_2 ] "w"(activations[2]), [ a_3 ] "w"(activations[3]),
        [ kernel_size ] "r"(conv_kernel_size), [ depth ] "r"(depth)
      : "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7");
}

/**
 * This function performs the accumulation loop when we know that the depth is
 * equal to one, i.e. the (bitpacked) number of input channels is 32.
 */
inline void AccumulationLoop_Depth1(std::int32_t conv_kernel_size,
                                    int32x4_t weights[2],
                                    int32x4_t activations[4],
                                    uint16x8_t accumulators[4],
                                    TBitpacked*& weights_ptr,
                                    TBitpacked** indirection_buffer) {
  ruy::profiler::ScopeLabel label("Accumulation loop (depth = 1)");

  // Declare these variables so we can use named registers in the ASM block.
  TBitpacked** indirection_buffer_ = indirection_buffer + 4;
  TBitpacked* a_ptr_0;
  TBitpacked* a_ptr_1;
  TBitpacked* a_ptr_2;
  TBitpacked* a_ptr_3;

  asm volatile(
      // Zero-out the accumulator registers
      "movi %[accum_0].8h, #0\n\t"
      "movi %[accum_1].8h, #0\n\t"
      "movi %[accum_2].8h, #0\n\t"
      "movi %[accum_3].8h, #0\n\t"

      "0:\n\t"  // Loop label

      "ldp %[a_ptr_0], %[a_ptr_1], [%[indirection_buffer]]\n\t"
      "ldp %[a_ptr_2], %[a_ptr_3], [%[indirection_buffer], #16]\n\t"
      "add %[indirection_buffer], %[indirection_buffer], #32\n\t"
      "subs %w[ks_index], %w[ks_index], #1\n\t"

      XOR_POPCOUNT_ACCUM_LOAD_BLOCK_32

      "bgt 0b\n\t"  // Loop branch
      : [ accum_0 ] "=&w"(accumulators[0]), [ accum_1 ] "=&w"(accumulators[1]),
        [ accum_2 ] "=&w"(accumulators[2]), [ accum_3 ] "=&w"(accumulators[3]),
        [ w_ptr ] "+r"(weights_ptr),
        [ indirection_buffer ] "+r"(indirection_buffer_),
        [ a_ptr_0 ] "=&r"(a_ptr_0), [ a_ptr_1 ] "=&r"(a_ptr_1),
        [ a_ptr_2 ] "=&r"(a_ptr_2), [ a_ptr_3 ] "=&r"(a_ptr_3),
        [ ks_index ] "+r"(conv_kernel_size)
      : [ w_0 ] "w"(weights[0]), [ w_1 ] "w"(weights[1]),
        [ a_0 ] "w"(activations[0]), [ a_1 ] "w"(activations[1]),
        [ a_2 ] "w"(activations[2]), [ a_3 ] "w"(activations[3])
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

#undef XOR_POPCOUNT_ACCUM_LOAD_BLOCK_32

/**
 * The output transform and store function for float output. Also reloads the
 * weight and activation registers for the next iteration.
 */
inline void OutputTransformAndLoadNextAndStore(
    const std::int32_t c_out_index, const std::int32_t channels_out,
    uint16x8_t accumulators[4], int32x4_t weights[2], int32x4_t activations[4],
    const bconv2d::OutputTransform<float>& output_transform,
    const TBitpacked* weights_ptr, TBitpacked** indirection_buffer,
    float*& output_ptr_0, float*& output_ptr_1, float*& output_ptr_2,
    float*& output_ptr_3) {
  ruy::profiler::ScopeLabel label("Float output transform and store");

  // Declare result registers.
  float32x4x2_t results[4];

  asm("ldp x0, x1, [%[indirection_buffer]]\n\t"

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

      "ldp x2, x3, [%[indirection_buffer], #16]\n\t"

      // Perform clamping
      "ldr q4, [%[bias_addr]]\n\t"
      "smax %[result_30].4s, %[result_30].4s, v0.4s\n\t"
      "smax %[result_31].4s, %[result_31].4s, v0.4s\n\t"
      "smax %[result_20].4s, %[result_20].4s, v0.4s\n\t"
      "smax %[result_21].4s, %[result_21].4s, v0.4s\n\t"
      "ldr q5, [%[bias_addr], #16]\n\t"
      "smax %[result_10].4s, %[result_10].4s, v0.4s\n\t"
      "smax %[result_11].4s, %[result_11].4s, v0.4s\n\t"
      "smax %[result_00].4s, %[result_00].4s, v0.4s\n\t"
      "smax %[result_01].4s, %[result_01].4s, v0.4s\n\t"
      "ld1r {%[a_0].4s}, [x0]\n\t"
      "smin %[result_30].4s, %[result_30].4s, v1.4s\n\t"
      "smin %[result_31].4s, %[result_31].4s, v1.4s\n\t"
      "smin %[result_20].4s, %[result_20].4s, v1.4s\n\t"
      "smin %[result_21].4s, %[result_21].4s, v1.4s\n\t"
      "ld1r {%[a_1].4s}, [x1]\n\t"
      "smin %[result_10].4s, %[result_10].4s, v1.4s\n\t"
      "smin %[result_11].4s, %[result_11].4s, v1.4s\n\t"
      "smin %[result_00].4s, %[result_00].4s, v1.4s\n\t"
      "smin %[result_01].4s, %[result_01].4s, v1.4s\n\t"

      // Convert to float, multiply by the multiplier, and add the bias. Note
      // that the float conversion instructions ("scvtf") are *very* slow and
      // block the Neon pipeline. It is therefore important for optimal
      // performance to interleave the float multiply and add instructions
      // between the "scvtf" instructions.
      "ld1r {%[a_2].4s}, [x2]\n\t"
      "scvtf %[result_30].4s, %[result_30].4s\n\t"
      "scvtf %[result_31].4s, %[result_31].4s\n\t"
      "ld1r {%[a_3].4s}, [x3]\n\t"
      "scvtf %[result_20].4s, %[result_20].4s\n\t"
      "fmul %[result_30].4s, %[result_30].4s, v2.4s\n\t"
      "scvtf %[result_21].4s, %[result_21].4s\n\t"
      "ldr %q[w_0], [%[w_ptr], #-32]\n\t"
      "fmul %[result_31].4s, %[result_31].4s, v3.4s\n\t"
      "fadd %[result_30].4s, %[result_30].4s, v4.4s\n\t"
      "scvtf %[result_10].4s, %[result_10].4s\n\t"
      "ldr %q[w_1], [%[w_ptr], #-16]\n\t"
      "fmul %[result_20].4s, %[result_20].4s, v2.4s\n\t"
      "fadd %[result_31].4s, %[result_31].4s, v5.4s\n\t"
      "scvtf %[result_11].4s, %[result_11].4s\n\t"
      "fmul %[result_21].4s, %[result_21].4s, v3.4s\n\t"
      "fadd %[result_20].4s, %[result_20].4s, v4.4s\n\t"
      "scvtf %[result_00].4s, %[result_00].4s\n\t"
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
      : [ w_0 ] "=&w"(weights[0]), [ w_1 ] "=&w"(weights[1]),
        [ a_0 ] "=&w"(activations[0]), [ a_1 ] "=&w"(activations[1]),
        [ a_2 ] "=&w"(activations[2]), [ a_3 ] "=&w"(activations[3]),
        [ result_00 ] "=&w"(results[0].val[0]),
        [ result_01 ] "=&w"(results[0].val[1]),
        [ result_10 ] "=&w"(results[1].val[0]),
        [ result_11 ] "=&w"(results[1].val[1]),
        [ result_20 ] "=&w"(results[2].val[0]),
        [ result_21 ] "=&w"(results[2].val[1]),
        [ result_30 ] "=&w"(results[3].val[0]),
        [ result_31 ] "=&w"(results[3].val[1])
      : [ accum_0 ] "w"(accumulators[0]), [ accum_1 ] "w"(accumulators[1]),
        [ accum_2 ] "w"(accumulators[2]), [ accum_3 ] "w"(accumulators[3]),
        [ clamp_min_addr ] "r"(&output_transform.clamp_min),
        [ clamp_max_addr ] "r"(&output_transform.clamp_max),
        [ multiplier_addr ] "r"(output_transform.multiplier + c_out_index),
        [ bias_addr ] "r"(output_transform.bias + c_out_index),
        [ w_ptr ] "r"(weights_ptr),
        [ indirection_buffer ] "r"(indirection_buffer)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5");

  if (channels_out - c_out_index >= 8) {
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
  } else {
#define STORE_SINGLE_ELEMENT_LOW(index)                           \
  vst1q_lane_f32(output_ptr_3 + index, results[3].val[0], index); \
  vst1q_lane_f32(output_ptr_2 + index, results[2].val[0], index); \
  vst1q_lane_f32(output_ptr_1 + index, results[1].val[0], index); \
  vst1q_lane_f32(output_ptr_0 + index, results[0].val[0], index);
#define STORE_SINGLE_ELEMENT_HIGH(index)                              \
  vst1q_lane_f32(output_ptr_3 + 4 + index, results[3].val[1], index); \
  vst1q_lane_f32(output_ptr_2 + 4 + index, results[2].val[1], index); \
  vst1q_lane_f32(output_ptr_1 + 4 + index, results[1].val[1], index); \
  vst1q_lane_f32(output_ptr_0 + 4 + index, results[0].val[1], index);
    STORE_SINGLE_ELEMENT_LOW(0)
    if (channels_out - c_out_index >= 2) {
      STORE_SINGLE_ELEMENT_LOW(1)
      if (channels_out - c_out_index >= 3) {
        STORE_SINGLE_ELEMENT_LOW(2)
        if (channels_out - c_out_index >= 4) {
          STORE_SINGLE_ELEMENT_LOW(3)
          if (channels_out - c_out_index >= 5) {
            STORE_SINGLE_ELEMENT_HIGH(0)
            if (channels_out - c_out_index >= 6) {
              STORE_SINGLE_ELEMENT_HIGH(1)
              if (channels_out - c_out_index >= 7) {
                STORE_SINGLE_ELEMENT_HIGH(2)
              }
            }
          }
        }
      }
    }
#undef STORE_SINGLE_ELEMENT_LOW
#undef STORE_SINGLE_ELEMENT_HIGH
  }
}

/**
 * The output transform and store function for int8 output. Also reloads the
 * weight and activation registers for the next iteration.
 */
inline void OutputTransformAndLoadNextAndStore(
    const std::int32_t c_out_index, const std::int32_t channels_out,
    uint16x8_t accumulators[4], int32x4_t weights[2], int32x4_t activations[4],
    const bconv2d::OutputTransform<std::int8_t>& output_transform,
    const TBitpacked* weights_ptr, TBitpacked** indirection_buffer,
    std::int8_t*& output_ptr_0, std::int8_t*& output_ptr_1,
    std::int8_t*& output_ptr_2, std::int8_t*& output_ptr_3) {
  ruy::profiler::ScopeLabel label("Int8 output transform and store");

  // Declare result registers. These are wider than we need for just the final
  // int8 values, which is necessary for intermediate results.
  int8x16x2_t results[4];

  asm("ldp x0, x1, [%[indirection_buffer]]\n\t"

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

      "ldp x2, x3, [%[indirection_buffer], #16]\n\t"

      // Perform clamping
      "ldr q4, [%[bias_addr]]\n\t"
      "smax %[result_30].4s, %[result_30].4s, v0.4s\n\t"
      "smax %[result_31].4s, %[result_31].4s, v0.4s\n\t"
      "smax %[result_20].4s, %[result_20].4s, v0.4s\n\t"
      "smax %[result_21].4s, %[result_21].4s, v0.4s\n\t"
      "ldr q5, [%[bias_addr], #16]\n\t"
      "smax %[result_10].4s, %[result_10].4s, v0.4s\n\t"
      "smax %[result_11].4s, %[result_11].4s, v0.4s\n\t"
      "smax %[result_00].4s, %[result_00].4s, v0.4s\n\t"
      "smax %[result_01].4s, %[result_01].4s, v0.4s\n\t"
      "ld1r {%[a_0].4s}, [x0]\n\t"
      "smin %[result_30].4s, %[result_30].4s, v1.4s\n\t"
      "smin %[result_31].4s, %[result_31].4s, v1.4s\n\t"
      "smin %[result_20].4s, %[result_20].4s, v1.4s\n\t"
      "smin %[result_21].4s, %[result_21].4s, v1.4s\n\t"
      "ld1r {%[a_1].4s}, [x1]\n\t"
      "smin %[result_10].4s, %[result_10].4s, v1.4s\n\t"
      "smin %[result_11].4s, %[result_11].4s, v1.4s\n\t"
      "smin %[result_00].4s, %[result_00].4s, v1.4s\n\t"
      "smin %[result_01].4s, %[result_01].4s, v1.4s\n\t"

      // Convert to float, multiply by the multiplier, add the bias, convert
      // back to integer, and use a series of "saturating extract narrow"
      // instructions to narrow the result to Int8. Note that the float
      // conversion instructions ("scvtf" and "fcvtns") are *very* slow and
      // block the Neon pipeline. It is therefore important for optimal
      // performance to interleave other instructions between them.
      "ld1r {%[a_2].4s}, [x2]\n\t"
      "scvtf %[result_30].4s, %[result_30].4s\n\t"
      "scvtf %[result_31].4s, %[result_31].4s\n\t"
      "ld1r {%[a_3].4s}, [x3]\n\t"
      "scvtf %[result_20].4s, %[result_20].4s\n\t"
      "fmul %[result_30].4s, %[result_30].4s, v2.4s\n\t"
      "scvtf %[result_21].4s, %[result_21].4s\n\t"
      "ldr %q[w_0], [%[w_ptr], #-32]\n\t"
      "fmul %[result_31].4s, %[result_31].4s, v3.4s\n\t"
      "fadd %[result_30].4s, %[result_30].4s, v4.4s\n\t"
      "scvtf %[result_10].4s, %[result_10].4s\n\t"
      "ldr %q[w_1], [%[w_ptr], #-16]\n\t"
      "fmul %[result_20].4s, %[result_20].4s, v2.4s\n\t"
      "fadd %[result_31].4s, %[result_31].4s, v5.4s\n\t"
      "scvtf %[result_11].4s, %[result_11].4s\n\t"
      "fmul %[result_21].4s, %[result_21].4s, v3.4s\n\t"
      "fadd %[result_20].4s, %[result_20].4s, v4.4s\n\t"
      "fcvtns %[result_30].4s, %[result_30].4s\n\t"
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
      : [ w_0 ] "=&w"(weights[0]), [ w_1 ] "=&w"(weights[1]),
        [ a_0 ] "=&w"(activations[0]), [ a_1 ] "=&w"(activations[1]),
        [ a_2 ] "=&w"(activations[2]), [ a_3 ] "=&w"(activations[3]),
        [ result_00 ] "=&w"(results[0].val[0]),
        [ result_01 ] "=&w"(results[0].val[1]),
        [ result_10 ] "=&w"(results[1].val[0]),
        [ result_11 ] "=&w"(results[1].val[1]),
        [ result_20 ] "=&w"(results[2].val[0]),
        [ result_21 ] "=&w"(results[2].val[1]),
        [ result_30 ] "=&w"(results[3].val[0]),
        [ result_31 ] "=&w"(results[3].val[1])
      : [ accum_0 ] "w"(accumulators[0]), [ accum_1 ] "w"(accumulators[1]),
        [ accum_2 ] "w"(accumulators[2]), [ accum_3 ] "w"(accumulators[3]),
        [ clamp_min_addr ] "r"(&output_transform.clamp_min),
        [ clamp_max_addr ] "r"(&output_transform.clamp_max),
        [ multiplier_addr ] "r"(output_transform.multiplier + c_out_index),
        [ bias_addr ] "r"(output_transform.bias + c_out_index),
        [ w_ptr ] "r"(weights_ptr),
        [ indirection_buffer ] "r"(indirection_buffer)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5");

  if (channels_out - c_out_index >= 8) {
    vst1_s8(output_ptr_3, vget_low_s8(results[3].val[0]));
    output_ptr_3 += 8;
    vst1_s8(output_ptr_2, vget_low_s8(results[2].val[0]));
    output_ptr_2 += 8;
    vst1_s8(output_ptr_1, vget_low_s8(results[1].val[0]));
    output_ptr_1 += 8;
    vst1_s8(output_ptr_0, vget_low_s8(results[0].val[0]));
    output_ptr_0 += 8;
  } else {
#define STORE_SINGLE_ELEMENT(index)                                          \
  vst1_lane_s8(output_ptr_3 + index, vget_low_s8(results[3].val[0]), index); \
  vst1_lane_s8(output_ptr_2 + index, vget_low_s8(results[2].val[0]), index); \
  vst1_lane_s8(output_ptr_1 + index, vget_low_s8(results[1].val[0]), index); \
  vst1_lane_s8(output_ptr_0 + index, vget_low_s8(results[0].val[0]), index);
    STORE_SINGLE_ELEMENT(0)
    if (channels_out - c_out_index >= 2) {
      STORE_SINGLE_ELEMENT(1)
      if (channels_out - c_out_index >= 3) {
        STORE_SINGLE_ELEMENT(2)
        if (channels_out - c_out_index >= 4) {
          STORE_SINGLE_ELEMENT(3)
          if (channels_out - c_out_index >= 5) {
            STORE_SINGLE_ELEMENT(4)
            if (channels_out - c_out_index >= 6) {
              STORE_SINGLE_ELEMENT(5)
              if (channels_out - c_out_index >= 7) {
                STORE_SINGLE_ELEMENT(6)
              }
            }
          }
        }
      }
    }
#undef STORE_SINGLE_ELEMENT
  }
}

/**
 * A 8x4x1 Neon micro-kernel for float or int8 output on Aarch64.
 */
template <typename DstScalar, bool Depth2OrMore>
void RunKernel(const std::int32_t block_num_pixels,
               const std::int32_t conv_kernel_size,
               const std::int32_t channels_in, const std::int32_t channels_out,
               const bconv2d::OutputTransform<DstScalar>& output_transform,
               const TBitpacked* weights_ptr_,
               const TBitpacked** indirection_buffer_, DstScalar* output_ptr) {
  // Correctness of this function relies on TBitpacked being four bytes.
  static_assert(sizeof(TBitpacked) == 4, "");

  ruy::profiler::ScopeLabel label(
      "Indirect BGEMM block (8x4x1, Aarch64 optimised)");

  TFLITE_DCHECK_GE(block_num_pixels, 1);
  TFLITE_DCHECK_LE(block_num_pixels, 4);
  TFLITE_DCHECK_GE(conv_kernel_size, 1);
  TFLITE_DCHECK_GE(channels_in, 1);
  TFLITE_DCHECK_GE(channels_out, 1);

  if (conv_kernel_size < 1 || channels_in < 1 || channels_out < 1) return;

  TBitpacked** indirection_buffer =
      const_cast<TBitpacked**>(indirection_buffer_);
  TBitpacked* weights_ptr = const_cast<TBitpacked*>(weights_ptr_);

  DstScalar* output_ptr_0 = output_ptr;
  DstScalar* output_ptr_1 = output_ptr + channels_out;
  DstScalar* output_ptr_2 = output_ptr + 2 * channels_out;
  DstScalar* output_ptr_3 = output_ptr + 3 * channels_out;

  // At the end of the output array we might get a block where the number of
  // pixels is less than 4. When this happens we set the 'leftover' output
  // pointer equal to the first output pointer, so that there's no risk of
  // writing beyond the array bounds. At the end we write to the output array
  // 'back to front' so that the outputs for the first pixel are written last,
  // which means that the result will still be correct.
  if (block_num_pixels < 4) {
    output_ptr_3 = output_ptr_0;
    if (block_num_pixels < 3) {
      output_ptr_2 = output_ptr_0;
      if (block_num_pixels < 2) {
        output_ptr_1 = output_ptr_0;
      }
    }
  }

  // Pre-load weights and activations.
  int32x4_t weights[2] = {vld1q_s32(weights_ptr), vld1q_s32(weights_ptr + 4)};
  weights_ptr += 8;
  int32x4_t activations[4] = {vld1q_dup_s32(indirection_buffer[0]),
                              vld1q_dup_s32(indirection_buffer[1]),
                              vld1q_dup_s32(indirection_buffer[2]),
                              vld1q_dup_s32(indirection_buffer[3])};

  std::int32_t c_out_index = 0;
  do {
    uint16x8_t accumulators[4];

    if (Depth2OrMore) {
      AccumulationLoop_Depth2OrMore(conv_kernel_size, channels_in, weights,
                                    activations, accumulators, weights_ptr,
                                    indirection_buffer);
    } else {
      AccumulationLoop_Depth1(conv_kernel_size, weights, activations,
                              accumulators, weights_ptr, indirection_buffer);
    }

    OutputTransformAndLoadNextAndStore(
        c_out_index, channels_out, accumulators, weights, activations,
        output_transform, weights_ptr, indirection_buffer, output_ptr_0,
        output_ptr_1, output_ptr_2, output_ptr_3);

    c_out_index += 8;
  } while (c_out_index < channels_out);
}

}  // namespace kernel_8x4x1_aarch64
}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_8x4x1_AARCH64_H_
