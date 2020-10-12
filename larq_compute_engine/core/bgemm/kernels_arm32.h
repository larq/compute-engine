#ifndef COMPUTE_ENGINE_CORE_BGEMM_KERNELS_ARM32_H_
#define COMPUTE_ENGINE_CORE_BGEMM_KERNELS_ARM32_H_

#include <cstdint>

#include "larq_compute_engine/core/bgemm/kernels_common.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"

using namespace ruy;

#if RUY_PLATFORM_NEON && RUY_OPT(ASM) && RUY_PLATFORM_NEON_32

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 4
#define RUY_OFFSET_DST_BASE_PTR 8
#define RUY_OFFSET_START_ROW 12
#define RUY_OFFSET_START_COL 16
#define RUY_OFFSET_LAST_ROW 20
#define RUY_OFFSET_LAST_COL 24
#define RUY_OFFSET_DST_ROWS 28
#define RUY_OFFSET_DST_COLS 32
#define RUY_OFFSET_LHS_STRIDE 36
#define RUY_OFFSET_RHS_STRIDE 40
#define RUY_OFFSET_DST_STRIDE 44
#define RUY_OFFSET_DEPTH 48
#define RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MIN 52
#define RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MAX 56
#define RUY_OFFSET_OUTPUT_TRANSFORM_MULTIPLIER 60
#define RUY_OFFSET_OUTPUT_TRANSFORM_BIAS 64

#define RUY_STACK_OFFSET_SIZE 96
#define RUY_STACK_OFFSET_DST_COL_PTR 0
#define RUY_STACK_OFFSET_DST_PTR 16
#define RUY_STACK_OFFSET_ROW 32
#define RUY_STACK_OFFSET_COL 48
#define RUY_STACK_OFFSET_LHS_COL_PTR 64
#define RUY_STACK_OFFSET_RHS_COL_PTR 80

template <typename Params>
void CheckOffsetsInKernelParams(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");

  const std::size_t OT_OFFSET = offsetof(Params, output_transform);

  // For float output
  static_assert(OT_OFFSET + offsetof(OutputTransform<float>, clamp_min) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MIN,
                "");
  static_assert(OT_OFFSET + offsetof(OutputTransform<float>, clamp_max) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MAX,
                "");
  static_assert(OT_OFFSET + offsetof(OutputTransform<float>, multiplier) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_MULTIPLIER,
                "");
  static_assert(OT_OFFSET + offsetof(OutputTransform<float>, bias) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_BIAS,
                "");

  // For int8 output
  static_assert(OT_OFFSET + offsetof(OutputTransform<std::int8_t>, clamp_min) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MIN,
                "");
  static_assert(OT_OFFSET + offsetof(OutputTransform<std::int8_t>, clamp_max) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MAX,
                "");
  static_assert(
      OT_OFFSET + offsetof(OutputTransform<std::int8_t>, multiplier) ==
          RUY_OFFSET_OUTPUT_TRANSFORM_MULTIPLIER,
      "");
  static_assert(OT_OFFSET + offsetof(OutputTransform<std::int8_t>, bias) ==
                    RUY_OFFSET_OUTPUT_TRANSFORM_BIAS,
                "");
}

#define MAKE_ZERO(reg) "vmov.s8 " #reg ", #0\n"

#define IF_FLOAT_OUTPUT(a) ".if %c[float_output]\n" a ".endif\n"

#define IF_INT8_OUTPUT(a) ".if %c[int8_output]\n" a ".endif\n"

#define IF_FLOAT_ELIF_INT8_OUTPUT(a, b) \
  ".if %c[float_output]\n" a ".elseif %c[int8_output]\n" b ".endif\n"

// clang-format off

#define LCE_BMLA(Vd, Vr, Vl0, Vl1, Vl2, Vl3) \
  "veor.s8 q12, " #Vr", " #Vl0 "\n"          \
  "veor.s8 q13, " #Vr", " #Vl1 "\n"          \
  "veor.s8 q14, " #Vr", " #Vl2 "\n"          \
  "veor.s8 q15, " #Vr", " #Vl3 "\n"          \
  "vcnt.s8 q12, q12\n"                       \
  "vcnt.s8 q13, q13\n"                       \
  "vcnt.s8 q14, q14\n"                       \
  "vcnt.s8 q15, q15\n"                       \
  "vpadd.i8 d24, d24, d25\n"                 \
  "vpadd.i8 d25, d26, d27\n"                 \
  "vpadd.i8 d28, d28, d29\n"                 \
  "vpadd.i8 d29, d30, d31\n"                 \
  "vpadd.i8 d24, d24, d25\n"                 \
  "vpadd.i8 d25, d28, d29\n"                 \
  "vpaddl.u8 q12, q12\n"                     \
  "vpadal.u16 " #Vd" , q12\n"

// clang-format on

// This is a very naive and first attempt on using the SIMD registers for BGEMM.
// The following optimizations still need to be implemented:
// 1. Using the entire register space which the architecture provides. This can
// be achieved in two ways:
// - 4x4 destination matrices and unrolling the depth loop
// - 8x8 destination matrices (requires dymanic changing of temporary
// registers in BMLA)
// 2. taking advantage of out-of-order cpu by dual dispatching the load/compute
// instructions

// The asm kernel below has the following NEON register allocation:
//
// v8, v9, v10, v11 are int32 accumulators.
// During accumulation, v0 -- v3 are used to load data from LHS and
// v4 -- v7 from RHS:
//
//                                      int32 RHS 4x4 block
//                          /--------------------------------------\
//                          |v4.s[0]         ...          v7.s[0]  |
//                          |  ...                         ...     |
//                          |v4.s[3]         ...          v7.s[3]  |
//                          \--------------------------------------/
//    int32 LHS 4x4 block
//  /--------------------\  /--------------------------------------\
//  |v0.s[0] ... v0.s[3] |  |v8.s[0]        ...         v11.s[0]  |
//  |v1.s[0] ... v1.s[3] |  |v8.s[1]        ...         v11.s[1]  |
//  |v2.s[0] ... v2.s[3] |  |v8.s[2]        ...         v11.s[2]  |
//  |v3.s[0] ... v3.s[3] |  |v8.s[3]        ...         v11.s[3]  |
//  \--------------------/  \--------------------------------------/
//                                  int32 accumulators 4x4 block
//
// No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
// optimization for this kernel.

template <typename DstScalar>
void BinaryKernelNeon4x4(BinaryKernelParams<DstScalar, 4, 4>& params) {
  CheckOffsetsInKernelParams(params);
  ruy::profiler::ScopeLabel label(
      "Binary Kernel (4x4) (kNeon, optimized for out-of-order cores)");

  static_assert(sizeof(TBitpacked) == 4,
                "Correctness of this function relies on the size of TBitpacked "
                "being 4 bytes.");

  TBitpacked* lhs_col_ptr = const_cast<TBitpacked*>(params.lhs_base_ptr);
  TBitpacked* rhs_col_ptr = const_cast<TBitpacked*>(params.rhs_base_ptr);
  TBitpacked* lhs_ptr = lhs_col_ptr;
  TBitpacked* rhs_ptr = rhs_col_ptr;

  DstScalar* dst_col_ptr = params.dst_base_ptr;
  DstScalar* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  asm volatile(
      "sub sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"
      // auto dst_col_ptr = params.dst_base_ptr;
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
      "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"

      // auto dst_ptr = dst_col_ptr;
      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
      "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

      // auto row = params.start_row
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
      "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

      // auto col = params.start_col
      "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_START_COL) "]\n"
      "str r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

      // auto lhs_col_ptr = params.lhs_base_ptr
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"

      // auto rhs_col_ptr = params.rhs_base_ptr
      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_RHS_BASE_PTR) "]\n"
      "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"

      // Clear accumulators.
      MAKE_ZERO(q8)
      MAKE_ZERO(q9)
      MAKE_ZERO(q10)
      MAKE_ZERO(q11)

      // Load the first 64 bytes of LHS and RHS data.
      "vld1.32 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5, d6, d7}, [%[lhs_ptr]]!\n"
      "vld1.32 {d8, d9, d10, d11}, [%[rhs_ptr]]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs_ptr]]!\n"

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for. The RHS is stored in col-wise. Therefore, for 32-bit elements,
      // one register can hold 4 levels of depth.
      "mov r1, #4\n"

      // Main loop of the whole GEMM, over rows and columns of the
      // destination matrix.
      "1:\n"

      LCE_BMLA(q8,  q4, q0, q1, q2, q3)
      LCE_BMLA(q9,  q5, q0, q1, q2, q3)
      LCE_BMLA(q10, q6, q0, q1, q2, q3)
      LCE_BMLA(q11, q7, q0, q1, q2, q3)

      // Accumulation loop
      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
      "cmp r1, r2\n"
      "beq 79f\n"

      "2:\n"

      "vld1.32 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5, d6, d7}, [%[lhs_ptr]]!\n"
      "vld1.32 {d8, d9, d10, d11}, [%[rhs_ptr]]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs_ptr]]!\n"

      "add r1, r1, #4\n"
      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
      "cmp r1, r2\n"

      LCE_BMLA(q8,  q4, q0, q1, q2, q3)
      LCE_BMLA(q9,  q5, q0, q1, q2, q3)
      LCE_BMLA(q10, q6, q0, q1, q2, q3)
      LCE_BMLA(q11, q7, q0, q1, q2, q3)

      "blt 2b\n"

      "79:\n"
      // End of accumulation. The registers v8 -- v11 contain the final
      // int32 accumulator values of the current 4x4 destination block.

      // Logic to advance to the next block in preparation for the next
      // iteration of the main loop. For now, we only want to compute
      // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
      // not yet ready to update the values of row and col, as we still need
      // the current values for the rest of the work on the current block.
      "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
      "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
      "cmp r1, r3\n"  // Have we finished the last row?
      "bge 4f\n"           // If finished last row, go to 4
      // Not finished last row: then advance to next row.
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
      "add r4, r4, r1, lsl #2\n"
      "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
      "b 5f\n"
      "4:\n"  // Finished last row...
      "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      // Go back to first row
      "str r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
      // Now we need to advance to the next column. If we already
      // finished the last column, then in principle we are done, however
      // we can't just return here, as we need to allow the end work of the
      // current block to complete. The good news is that at this point it
      // doesn't matter what data we load for the next column, since
      // we will exit from the main loop below before actually storing
      // anything computed from that data.
      "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
      "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
      "cmp r8, r4\n"  // Have we finished the last column?
      "bge 5f\n" // If yes, just carry on without updating the column pointer.
      // Not finished last column: then advance to next column.
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
      "ldr r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
      "add r10, r10, r1, lsl #2\n"
      "str r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
      "5:\n"

      // Set the LHS and RHS data pointers to the start of the columns just
      // computed.
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
      "mov %[lhs_ptr], r4\n"
      "ldr r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
      "mov %[rhs_ptr], r5\n"

      // post-accumulation transformation:
      //
      // q13 = |BKADD|BKADD|BKADD|BKADD|
      // q14 = |MULT0|MULT1|MULT2|MULT3|
      // q15 = |BIAS0|BIAS1|BIAS2|BIAS3|
      //

      // Load multiplication bias
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_OUTPUT_TRANSFORM_MULTIPLIER) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
      "add r1, r1, r8, lsl #2\n"
      // Load 4 bias-multiplication values.
      "vld1.32 {d28, d29}, [r1]!\n"

      // Load addition bias, r8 still holds "row offset"
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_OUTPUT_TRANSFORM_BIAS) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "add r1, r1, r8, lsl #2\n"
      // Load 4 bias-addition values.
      "vld1.32 {d30, d31}, [r1]!\n"

      // Now that we know what LHS and RHS data the next iteration of the
      // main loop will need to load, we start loading the first 64 bytes of
      // each of LHS and RHS, into v0 -- v3 and v4 -- v7 as we don't need
      // them anymore in the rest of the work on the current block.
      "vld1.32 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5, d6, d7}, [%[lhs_ptr]]!\n"
      "vld1.32 {d8, d9, d10, d11}, [%[rhs_ptr]]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs_ptr]]!\n"

      // Load the clamp_max bound (in parallel with the shift)
      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MIN) "]\n"
      "vdup q12.32, r1 \n"  // clamp_min

      // Perform the backtransformation shift (in int32)
      "vshl q8.s32, q8.s32, #1\n"
      "vshl q9.s32, q9.s32, #1\n"
      "vshl q10.s32, q10.s32, #1\n"
      "vshl q11.s32, q11.s32, #1\n"

      // Load the clamp_max bound (in parallel with the clamp_min)
      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MAX) "]\n"
      "vdup q13.32, r2\n"  // clamp_max

      // Perform the activation function, by clamping
      // Apply the clamp_min bound
      "vmax q8.s32, q8.s32, q12.s32\n"
      "vmax q9.s32, q9.s32, q12.s32\n"
      "vmax q10.s32, q10.s32, q12.s32\n"
      "vmax q11.s32, q11.s32, q12.s32\n"
      // Apply the clamp_max bound
      "vmin q8.s32, q8.s32, q13.s32\n"
      "vmin q9.s32, q9.s32, q13.s32\n"
      "vmin q10.s32, q10.s32, q13.s32\n"
      "vmin q11.s32, q11.s32, q13.s32\n"

      // Convert to single precision float
      "vcvt.f32.s32 q8, q8\n"
      "vcvt.f32.s32 q9, q9\n"
      "vcvt.f32.s32 q10, q10\n"
      "vcvt.f32.s32 q11, q11\n"

      // Perform the post multiplications
      "vmul.f32 q8, q8, q14\n"
      "vmul.f32 q9, q9, q14\n"
      "vmul.f32 q10, q10, q14\n"
      "vmul.f32 q11, q11, q14\n"

      // Perform the post additions
      "vadd.f32 q8, q8, q15\n"
      "vadd.f32 q9, q9, q15\n"
      "vadd.f32 q10, q10, q15\n"
      "vadd.f32 q11, q11, q15\n"

IF_INT8_OUTPUT(
      // Convert to signed integer, rounding to nearest.
      //     Arm32 NEON has an instruction (vcvtr) to convert from float->int
      // with rounding-to-nearest behaviour, but for some reason when used here
      // truncation (rounding-to-zero) behaviour was observed instead. It's
      // unclear what the issue is - perhaps QEMU doesn't correctly emulate the
      // rounding behaviour?
      //     The work-around is to first convert from float to fixed-point (24
      // integer bits, 8 fractional bits)...
      "vcvt.s32.f32 q8, q8, #8\n"
      "vcvt.s32.f32 q9, q9, #8\n"
      "vcvt.s32.f32 q10, q10, #8\n"
      "vcvt.s32.f32 q11, q11, #8\n"
      // ...and then use "vector saturating shift right and narrow, by immediate
      // value, with rounding" instructions to round to the nearest integer and
      // simultaneously narrow the result to int16.
      "vqrshrn.s32 d16, q8, #8\n"
      "vqrshrn.s32 d18, q9, #8\n"
      "vqrshrn.s32 d20, q10, #8\n"
      "vqrshrn.s32 d22, q11, #8\n"

      // Narrow to int8 with "signed saturating extract narrow" instructions.
      "vqmovn.s16 d16, q8\n"
      "vqmovn.s16 d18, q9\n"
      "vqmovn.s16 d20, q10\n"
      "vqmovn.s16 d22, q11\n"
)

      // Done post-accumuation transformation.

      // Compute how much of the 4x4 block of destination values that
      // we have computed, fit in the destination matrix. Typically, all of
      // it fits, but when the destination matrix shape is not a multiple
      // of 4x4, there are some 4x4 blocks along the boundaries that do
      // not fit entirely.

      "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
      "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
      "sub r1, r1, r8\n"

      "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
      "sub r2, r2, r4\n"

      "mov r3, #4\n"
      "mov r5, #4\n"

      "cmp r1, #4\n"
      // Compute r1 = how many rows of the 4x4 block fit
      "it gt\n"
      "movgt r1, r3\n"

      // Compute r2 = how many cols of the 4x4 block fit
      "cmp r2, #4\n"
      "it gt\n"
      "movgt r2, r5\n"

      // Test if r1==4 && r2 == 4, i.e. if all of the 4x4 block fits.
      "cmp r1, r3\n"
      "it eq\n"
      "cmpeq r2, r5\n"
      // Yes, all of the 4x4 block fits, go to fast path.
      "beq 30f\n"
      // Not all of the 4x4 block fits.
      // Set (r3 address, r4 stride) to write to dst_tmp_buf
      "mov r3, %[dst_tmp_buf]\n"
IF_FLOAT_ELIF_INT8_OUTPUT(
      "mov r4, #16\n"
,
      "mov r4, #4\n"
)
      "b 31f\n"
      "30:\n"
      // Yes, all of the 4x4 block fits.
      // Set (x3 address, x4 stride) to write directly to destination matrix.
      "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
      "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
      "31:\n"

      // Write our values to the destination described by
      // (x3 address, x4 stride).
IF_FLOAT_ELIF_INT8_OUTPUT(
      "vst1.32 {d16, d17}, [r3], r4\n"
      "vst1.32 {d18, d19}, [r3], r4\n"
      MAKE_ZERO(q8)
      MAKE_ZERO(q9)
      "vst1.32 {d20, d21}, [r3], r4\n"
      "vst1.32 {d22, d23}, [r3]\n"
      MAKE_ZERO(q10)
      MAKE_ZERO(q11)
,
      "vst1.32 {d16[0]}, [r3], r4\n"
      "vst1.32 {d18[0]}, [r3], r4\n"
      MAKE_ZERO(q8)
      MAKE_ZERO(q9)
      "vst1.32 {d20[0]}, [r3], r4\n"
      "vst1.32 {d22[0]}, [r3]\n"
      MAKE_ZERO(q10)
      MAKE_ZERO(q11)
)

      // If all of the 4x4 block fits, we just finished writing it to the
      // destination, so we skip the next part.
      "beq 41f\n"
      // Not all of the 4x4 block fits in the destination matrix.  We just
      // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
      // it to copy into the destination matrix the part that fits.

      "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
      "mov r3, %[dst_tmp_buf]\n"
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
      "mov r6, #0\n"
      "50:\n"
      "mov r5, #0\n"
      "51:\n"

IF_FLOAT_ELIF_INT8_OUTPUT(
      "ldr r10, [r3, r5, lsl #2]\n"
      "str r10, [r4, r5, lsl #2]\n"
,
      "ldrb r10, [r3, r5]\n"
      "strb r10, [r4, r5]\n"
)
      "add r5, r5, #1\n"
      "cmp r5, r1\n"
      "blt 51b\n"
      "add r6, r6, #1\n"
IF_FLOAT_ELIF_INT8_OUTPUT(
      "add r3, r3, #16\n"
,
      "add r3, r3, #4\n"
)
      "add r4, r4, r8\n"
      "cmp r6, r2\n"
      "blt 50b\n"
      "41:\n"
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
IF_FLOAT_ELIF_INT8_OUTPUT(
      "add r4, r4, #16\n"
,
      "add r4, r4, #4\n"
)
      "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

      // At this point we have completely finished writing values to the
      // destination matrix for the current block.

      // Reload some params --- we had used r3, r5, r10 for a few other things
      // since the last time we had loaded them.
      "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      "ldr r6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
      "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

      // Move to the next block of the destination matrix, for the next iter
      // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
      // been updated earlier.
      // Have we reached the end row?
      "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
      "cmp r8, r3\n"

      "beq 20f\n"  // yes, end row.
      // Not end row. Move to the next row.
      "add r8, r8, #4\n"
      // Store new value of row
      "str r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

      "b 21f\n"
      "20:\n"
      // Was already at end row.
      // Move back to first row.
      "str r6, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
      // Move to the next column.
      "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
      "add r4, r4, #4\n"
      "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

      "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
      "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
      // Increment dst_col_ptr by 4 * dst_stride (i.e. 4 columns)
      "add r1, r1, r8, lsl #2\n"
      // Store dst_col_ptr
      "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
      // Store dst_ptr
      "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
      "21:\n"

      // Main loop exit condition: have we hit the end column?
      "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
      "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
       "cmp r8, r4\n"

      // r1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for.
      "mov r1, #4\n"

      "ble 1b\n"

      "add sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

      : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
      : [ params ] "r"(&params), [dst_tmp_buf] "r"(params.dst_tmp_buf),
        [float_output]"i"(std::is_same<DstScalar, float>::value),
        [int8_output]"i"(std::is_same<DstScalar, std::int8_t>::value)
      : "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "cc",
        "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
        "q13", "q14", "q15" );
}

#undef MAKE_ZERO
#undef IF_FLOAT_OUTPUT
#undef IF_INT8_OUTPUT
#undef IF_FLOAT_ELIF_INT8_OUTPUT

#undef RUY_STACK_OFFSET_SIZE
#undef RUY_STACK_OFFSET_DST_COL_PTR
#undef RUY_STACK_OFFSET_DST_PTR
#undef RUY_STACK_OFFSET_ROW
#undef RUY_STACK_OFFSET_COL
#undef RUY_STACK_OFFSET_LHS_COL_PTR
#undef RUY_STACK_OFFSET_RHS_COL_PTR

#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_DST_ROWS
#undef RUY_OFFSET_DST_COLS
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_OUTPUT_TRANSFORM_MULTIPLIER
#undef RUY_OFFSET_OUTPUT_TRANSFORM_BIAS
#undef RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MIN
#undef RUY_OFFSET_OUTPUT_TRANSFORM_CLAMP_MAX

#endif  // RUY_PLATFORM_NEON && RUY_OPT(ASM) && RUY_PLATFORM_NEON_32

#endif  // COMPUTE_ENGINE_CORE_BGEMM_KERNELS_ARM32_H_
