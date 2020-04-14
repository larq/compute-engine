#include <cstdint>

#include "bgemm_kernels_common.h"
#include "profiling/instrumentation.h"

using namespace ruy;

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// clang-format off

// temporary NEON registers: v28,v29,v30
#define LCE_BMLA(Vd, Vr, Vl1, Vl2, Vl3, Vl4) \
  "eor v28.16b, " #Vr".16b, " #Vl1".16b\n"    \
  "eor v29.16b, " #Vr".16b, " #Vl2".16b\n"    \
  "cnt v28.16b, v28.16b\n"                    \
  "cnt v29.16b, v29.16b\n"                    \
  "addp v28.16b, v28.16b, v29.16b\n"          \
  "eor v29.16b, " #Vr".16b, " #Vl3".16b\n"    \
  "eor v30.16b, " #Vr".16b, " #Vl4".16b\n"    \
  "cnt v29.16b, v29.16b\n"                    \
  "cnt v30.16b, v30.16b\n"                    \
  "addp v29.16b, v29.16b, v30.16b\n"          \
  "addp v28.16b, v28.16b, v29.16b\n"          \
  "uaddlp v28.8h, v28.16b\n"                  \
  "uadalp " #Vd".4s, v28.8h\n"

// temporary NEON registers: v28,v29,v30
#define LCE_BMLA_LD_RHS(Vd, Vr, Vl1, Vl2, Vl3, Vl4)      \
  "eor v28.16b, " #Vr".16b, " #Vl1".16b\n"              \
  "eor v29.16b, " #Vr".16b, " #Vl2".16b\n"              \
  "cnt v28.16b, v28.16b\n"                              \
  "cnt v29.16b, v29.16b\n"                              \
  "addp v28.16b, v28.16b, v29.16b\n"                    \
  "eor v29.16b, " #Vr".16b, " #Vl3".16b\n"              \
  "eor v30.16b, " #Vr".16b, " #Vl4".16b\n"              \
  "cnt v29.16b, v29.16b\n"                              \
  "cnt v30.16b, v30.16b\n"                              \
  "ld1 {"#Vr".2d}, [%[rhs_ptr]], #16\n"                 \
  "addp v29.16b, v29.16b, v30.16b\n"                    \
  "addp v28.16b, v28.16b, v29.16b\n"                    \
  "uaddlp v28.8h, v28.16b\n"                            \
  "uadalp " #Vd".4s, v28.8h\n"

// temporary NEON registers: v28,v29,v30
#define LCE_BMLA_LD_ALL(Vd, Vr, Vl1, Vl2, Vl3, Vl4)      \
  "eor v28.16b, " #Vr".16b, " #Vl1".16b\n"              \
  "eor v29.16b, " #Vr".16b, " #Vl2".16b\n"              \
  "cnt v28.16b, v28.16b\n"                              \
  "cnt v29.16b, v29.16b\n"                              \
  "ld1 {"#Vl1".2d}, [%[lhs_ptr]], #16\n"                \
  "addp v28.16b, v28.16b, v29.16b\n"                    \
  "eor v29.16b, " #Vr".16b, " #Vl3".16b\n"              \
  "eor v30.16b, " #Vr".16b, " #Vl4".16b\n"              \
  "ld1 {"#Vl2".2d}, [%[lhs_ptr]], #16\n"                \
  "cnt v29.16b, v29.16b\n"                              \
  "cnt v30.16b, v30.16b\n"                              \
  "ld1 {"#Vr".2d}, [%[rhs_ptr]], #16\n"                 \
  "addp v29.16b, v29.16b, v30.16b\n"                    \
  "addp v28.16b, v28.16b, v29.16b\n"                    \
  "ld1 {"#Vl3".2d}, [%[lhs_ptr]], #16\n"                \
  "uaddlp v28.8h, v28.16b\n"                            \
  "uadalp " #Vd".4s, v28.8h\n"                          \
  "ld1 {"#Vl4".2d}, [%[lhs_ptr]], #16\n"

// clang-format on

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 8
#define RUY_OFFSET_DST_BASE_PTR 16
#define RUY_OFFSET_POST_ACTIVATION_MULTIPLIER 24
#define RUY_OFFSET_POST_ACTIVATION_BIAS 32
#define RUY_OFFSET_START_ROW 40
#define RUY_OFFSET_START_COL 44
#define RUY_OFFSET_LAST_ROW 48
#define RUY_OFFSET_LAST_COL 52
#define RUY_OFFSET_LHS_STRIDE 64
#define RUY_OFFSET_RHS_STRIDE 68
#define RUY_OFFSET_DST_STRIDE 72
#define RUY_OFFSET_DEPTH 76
#define RUY_OFFSET_CLAMP_MIN 80
#define RUY_OFFSET_CLAMP_MAX 84
#define RUY_OFFSET_BACKTRANSFORM_ADD 88
#define RUY_OFFSET_FLAGS 92

template <typename Params>
void CheckOffsetsInKernelParams32BP(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, post_activation_multiplier) ==
                    RUY_OFFSET_POST_ACTIVATION_MULTIPLIER,
                "");
  static_assert(
      offsetof(Params, post_activation_bias) == RUY_OFFSET_POST_ACTIVATION_BIAS,
      "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(
      offsetof(Params, backtransform_add) == RUY_OFFSET_BACKTRANSFORM_ADD, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

// This is a very naive and first attempt on using the SIMD registers for BGEMM.
// The following optimizations still need to be implemented:
// 1. Using the entire register space which the architecture provides. This can
// be achieved in two ways:
// - 4x4 destination matrices and unrolling the depth loop
// - 8x8 destination matrices (requires dymanic changing of temporary
// registers in BMLA)
// 2. taking advantage of out-of-order cpu by dual dispatching the load/compute
// instructions

// clang-format off

// The asm kernel below has the following NEON register allocation:
//
// v16, v18, v20, v22 are int32 accumulators.
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
//  |v0.s[0] ... v0.s[3] |  |v16.s[0]        ...         v22.s[0]  |
//  |v1.s[0] ... v1.s[3] |  |v16.s[1]        ...         v22.s[1]  |
//  |v2.s[0] ... v2.s[3] |  |v16.s[2]        ...         v22.s[2]  |
//  |v3.s[0] ... v3.s[3] |  |v16.s[3]        ...         v22.s[3]  |
//  \--------------------/  \--------------------------------------/
//                                  int32 accumulators 4x4 block
//
// No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
// optimization for this kernel.

// clang-format on

void BinaryKernelNeonOutOfOrder32BP4x4(
    const BinaryKernelParams<4, 4, std::uint32_t>& params) {
  CheckOffsetsInKernelParams32BP(params);
  gemmlowp::ScopedProfilingLabel label(
      "Binary Kernel (4x4) 32BP (kNeon, optimized for out-of-order cores)");

  std::uint32_t* lhs_col_ptr = const_cast<std::uint32_t*>(params.lhs_base_ptr);
  std::uint32_t* rhs_col_ptr = const_cast<std::uint32_t*>(params.rhs_base_ptr);
  std::uint32_t* lhs_ptr = lhs_col_ptr;
  std::uint32_t* rhs_ptr = rhs_col_ptr;

  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

      // clang-format off

      // Load some parameters into registers.
      "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
      "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
      "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
      "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
      "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
      "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
      "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

      // Load the first 64 bytes of LHS and RHS data.
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[lhs_ptr]], #64\n"
      "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[rhs_ptr]], #64\n"

      // Clear accumulators.
      RUY_MAKE_ZERO(v16)
      RUY_MAKE_ZERO(v18)
      RUY_MAKE_ZERO(v20)
      RUY_MAKE_ZERO(v22)

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for. The RHS is stored in col-wise. Therefore, for 32-bit elements,
      // one register can hold 4 levels of depth.
      "mov w1, #4\n"

      // Main loop of the whole GEMM, over rows and columns of the
      // destination matrix.
      "1:\n"

      LCE_BMLA(v16, v4, v0, v1, v2, v3)
      LCE_BMLA(v18, v5, v0, v1, v2, v3)
      LCE_BMLA(v20, v6, v0, v1, v2, v3)
      LCE_BMLA(v22, v7, v0, v1, v2, v3)

      // Accumulation loop
      "cmp w1, w12\n"
      "beq 79f\n"

      "2:\n"
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[lhs_ptr]], #64\n"
      "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[rhs_ptr]], #64\n"

      "add w1, w1, #4\n"
      "cmp w1, w12\n"

      LCE_BMLA(v16, v4, v0, v1, v2, v3)
      LCE_BMLA(v18, v5, v0, v1, v2, v3)
      LCE_BMLA(v20, v6, v0, v1, v2, v3)
      LCE_BMLA(v22, v7, v0, v1, v2, v3)

      "blt 2b\n"

      "79:\n"

      // End of accumulation. The registers v16 -- v22 contain the final
      // int32 accumulator values of the current 4x4 destination block.

      // Logic to advance to the next block in preparation for the next
      // iteration of the main loop. For now, we only want to compute
      // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
      // not yet ready to update the values of row and col, as we still need
      // the current values for the rest of the work on the current block.

      "cmp %w[row], w7\n"  // Have we finished the last row?
      "bge 4f\n"           // If finished last row, go to 4
      // Not finished last row: then advance to next row.
      // x9 is the LHS stride
      "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
      "b 5f\n"
      "4:\n"  // Finished last row...
      "mov %[lhs_col_ptr], x5\n"  // Go back to first row
      // Now we need to advance to the next column. If we already
      // finished the last column, then in principle we are done, however
      // we can't just return here, as we need to allow the end work of the
      // current block to complete. The good news is that at this point it
      // doesn't matter what data we load for the next column, since
      // we will exit from the main loop below before actually storing
      // anything computed from that data.
      "cmp %w[col], w8\n"  // Have we finished the last column?
      "bge 5f\n" // If yes, just carry on without updating the column pointer.
      // Not finished last column: then advance to next column.
      // x10 is the RHS stride
      "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
      "5:\n"

      // Set the LHS and RHS data pointers to the start of the columns just
      // computed.
      "mov %[lhs_ptr], %[lhs_col_ptr]\n"
      "mov %[rhs_ptr], %[rhs_col_ptr]\n"

      // Load some parameters needed for the end work on current block.
      "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"

      // Load backtransform add (duplicate 4 times into v13)
      "ldr w1, [%[params], #" RUY_STR(RUY_OFFSET_BACKTRANSFORM_ADD) "]\n"
      "dup v13.4s, w1 \n"

      // Load multiplication bias
      "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_POST_ACTIVATION_MULTIPLIER) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "add x2, x1, %x[row], lsl #2\n"
      "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
      "csel x1, x1, x2, eq\n"
      // Load 4 bias-multiplication values.
      "ld1 {v14.4s}, [x1], #16\n"

      // Load addition bias
      "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_POST_ACTIVATION_BIAS) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "add x2, x1, %x[row], lsl #2\n"
      "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
      "csel x1, x1, x2, eq\n"
      // Load 4 bias-addition values.
      "ld1 {v15.4s}, [x1], #16\n"

      // Now that we know what LHS and RHS data the next iteration of the
      // main loop will need to load, we start loading the first 64 bytes of
      // each of LHS and RHS, into v0 -- v3 and v4 -- v7 as we don't need
      // them anymore in the rest of the work on the current block.
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[lhs_ptr]], #64\n"
      "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[rhs_ptr]], #64\n"

      // Perform the backtransformation (in int32)
      "shl v16.4s, v16.4s, #1\n"
      "shl v18.4s, v18.4s, #1\n"
      "shl v20.4s, v20.4s, #1\n"
      "shl v22.4s, v22.4s, #1\n"

      // Load the clamp_max bound (in parallel with the sub)
      "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
      "dup v12.4s, w2\n"  // clamp_min

      "sub v16.4s, v13.4s, v16.4s\n"
      "sub v18.4s, v13.4s, v18.4s\n"
      "sub v20.4s, v13.4s, v20.4s\n"
      "sub v22.4s, v13.4s, v22.4s\n"

      // Load the clamp_max bound (in parallel with the clamp_min)
      "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
      "dup v13.4s, w3\n"  // clamp_max

      // Perform the activation function, by clamping
      // Apply the clamp_min bound
      "smax v16.4s, v16.4s, v12.4s\n"
      "smax v18.4s, v18.4s, v12.4s\n"
      "smax v20.4s, v20.4s, v12.4s\n"
      "smax v22.4s, v22.4s, v12.4s\n"
      // Apply the clamp_max bound
      "smin v16.4s, v16.4s, v13.4s\n"
      "smin v18.4s, v18.4s, v13.4s\n"
      "smin v20.4s, v20.4s, v13.4s\n"
      "smin v22.4s, v22.4s, v13.4s\n"

      // Convert to single precision float
      "scvtf v16.4s, v16.4s\n"
      "scvtf v18.4s, v18.4s\n"
      "scvtf v20.4s, v20.4s\n"
      "scvtf v22.4s, v22.4s\n"

      // Perform the post multiplications
      "fmul v16.4s, v16.4s, v14.4s\n"
      "fmul v18.4s, v18.4s, v14.4s\n"
      "fmul v20.4s, v20.4s, v14.4s\n"
      "fmul v22.4s, v22.4s, v14.4s\n"
      // Perform the post additions
      "fadd v16.4s, v16.4s, v15.4s\n"
      "fadd v18.4s, v18.4s, v15.4s\n"
      "fadd v20.4s, v20.4s, v15.4s\n"
      "fadd v22.4s, v22.4s, v15.4s\n"

      // Compute how much of the 4x4 block of destination values that
      // we have computed, fit in the destination matrix. Typically, all of
      // it fits, but when the destination matrix shape is not a multiple
      // of 4x4, there are some 4x4 blocks along the boundaries that do
      // not fit entirely.
      "sub w1, %w[dst_rows], %w[row]\n"
      "sub w2, %w[dst_cols], %w[col]\n"
      "mov w3, #4\n"
      "cmp w1, #4\n"
      // Compute w1 = how many rows of the 4x4 block fit
      "csel w1, w1, w3, le\n"
      "cmp w2, #4\n"
      // Compute w2 = how many cols of the 4x4 block fit
      "csel w2, w2, w3, le\n"

      // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
      "cmp w1, w3\n"
      "ccmp w2, w3, 0, eq\n"
      // Yes, all of the 4x4 block fits, go to fast path.
      "beq 30f\n"
      // Not all of the 4x4 block fits.
      // Set (x3 address, x4 stride) to write to dst_tmp_buf
      "mov x3, %[dst_tmp_buf]\n"
      "mov x4, #16\n"
      "b 31f\n"
      "30:\n"
      // Yes, all of the 4x4 block fits.
      // Set (x3 address, x4 stride) to write directly to destination matrix.
      "mov x3, %[dst_ptr]\n"
      "mov x4, x11\n"
      "31:\n"

      // Write our values to the destination described by
      // (x3 address, x4 stride).
      "str q16, [x3, #0]\n"
      "add x3, x3, x4\n"
      "str q18, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v16)
      RUY_MAKE_ZERO(v18)
      "str q20, [x3, #0]\n"
      "add x3, x3, x4\n"
      "str q22, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v20)
      RUY_MAKE_ZERO(v22)

      // If all of the 4x4 block fits, we just finished writing it to the
      // destination, so we skip the next part.
      "beq 41f\n"
      // Not all of the 4x4 block fits in the destination matrix.  We just
      // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
      // it to copy into the destination matrix the part that fits.
      "mov x3, %[dst_tmp_buf]\n"
      "mov x4, %[dst_ptr]\n"
      "mov w14, #0\n"
      "50:\n"
      "mov w15, #0\n"
      "51:\n"
      "ldr w13, [x3, x15, lsl #2]\n"
      "str w13, [x4, x15, lsl #2]\n"
      "add w15, w15, #1\n"
      "cmp w15, w1\n"
      "blt 51b\n"
      "add w14, w14, #1\n"
      "add x3, x3, #16\n"
      "add x4, x4, x11\n"
      "cmp w14, w2\n"
      "blt 50b\n"
      "41:\n"
      "add %[dst_ptr], %[dst_ptr], #16\n"

      // At this point we have completely finished writing values to the
      // destination matrix for the current block.

      // Move to the next block of the destination matrix, for the next iter
      // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
      // been updated earlier.
      // Have we reached the end row?
      "cmp %w[row], w7\n"
      "beq 20f\n"  // yes, end row.
      // Not end row. Move to the next row.
      "add %w[row], %w[row], #4\n"
      "b 21f\n"
      "20:\n"
      // Was already at end row.
      "mov %w[row], w6\n"  // Move back to first row.
      "add %w[col], %w[col], #4\n"  // Move to the next column.
      "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
      "mov %[dst_ptr], %[dst_col_ptr]\n"
      "21:\n"

      // Main loop exit condition: have we hit the end column?
      "cmp %w[col], w8\n"

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for.
      "mov w1, #4\n"

      "ble 1b\n"

      // clang-format on

      : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
      : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
        [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
      : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "cc",
        "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
}

#undef RUY_OFFSET_BACKTRANSFORM_ADD
#undef RUY_OFFSET_POST_ACTIVATION_MULTIPLIER
#undef RUY_OFFSET_POST_ACTIVATION_BIAS
#undef RUY_OFFSET_FLAGS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 8
#define RUY_OFFSET_DST_BASE_PTR 16
#define RUY_OFFSET_POST_ACTIVATION_MULTIPLIER 24
#define RUY_OFFSET_POST_ACTIVATION_BIAS 32
#define RUY_OFFSET_START_ROW 40
#define RUY_OFFSET_START_COL 44
#define RUY_OFFSET_LAST_ROW 48
#define RUY_OFFSET_LAST_COL 52
#define RUY_OFFSET_LHS_STRIDE 64
#define RUY_OFFSET_RHS_STRIDE 68
#define RUY_OFFSET_DST_STRIDE 72
#define RUY_OFFSET_DEPTH 76
#define RUY_OFFSET_CLAMP_MIN 80
#define RUY_OFFSET_CLAMP_MAX 84
#define RUY_OFFSET_BACKTRANSFORM_ADD 88
#define RUY_OFFSET_FLAGS 92

template <typename Params>
void CheckOffsetsInKernelParams64BP(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, post_activation_multiplier) ==
                    RUY_OFFSET_POST_ACTIVATION_MULTIPLIER,
                "");
  static_assert(
      offsetof(Params, post_activation_bias) == RUY_OFFSET_POST_ACTIVATION_BIAS,
      "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(
      offsetof(Params, backtransform_add) == RUY_OFFSET_BACKTRANSFORM_ADD, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

// clang-format off

// The asm kernel below has the following NEON register allocation:
//
// v24 -- v27 are int32 accumulators.
// During accumulation, v0 -- v3 are used to load data from LHS and
// v4 -- v7 from RHS:
//
//                                    int64 RHS 2x4 block
//                          /--------------------------------------\
//                          |v4.d[0]         ...          v7.d[0]  |
//                          |v4.d[1]         ...          v7.d[1]  |
//                          \--------------------------------------/
//     int64 LHS 4x2 block
//      /----------------\  /--------------------------------------\
//      |v0.d[0] v0.d[1] |  |v24.s[0]        ...         v27.s[0]  |
//      |v1.d[0] v1.d[1] |  |v24.s[1]        ...         v27.s[1]  |
//      |v2.d[0] v2.d[1] |  |v24.s[2]        ...         v27.s[2]  |
//      |v3.d[0] v3.d[1] |  |v24.s[3]        ...         v27.s[3]  |
//      \----------------/  \--------------------------------------/
//                                  int32 accumulators 4x4 block
//
// In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
// is repeated 2 times, using 2x more registers for LHS and RHS.

// clang-format on

void BinaryKernelNeonOutOfOrder64BP4x4(
    const BinaryKernelParams<4, 4, std::uint64_t>& params) {
  CheckOffsetsInKernelParams64BP(params);
  gemmlowp::ScopedProfilingLabel label(
      "Binary Kernel (4x4) 64BP (kNeon, optimized for out-of-order cores)");

  std::uint64_t* lhs_col_ptr = const_cast<std::uint64_t*>(params.lhs_base_ptr);
  std::uint64_t* rhs_col_ptr = const_cast<std::uint64_t*>(params.rhs_base_ptr);
  std::uint64_t* lhs_ptr = lhs_col_ptr;
  std::uint64_t* rhs_ptr = rhs_col_ptr;

  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

      // clang-format off

      // Load some parameters into registers.
      "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
      RUY_MAKE_ZERO(v24)
      "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
      "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
      RUY_MAKE_ZERO(v25)
      "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
      "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
      RUY_MAKE_ZERO(v26)
      "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
      "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
      RUY_MAKE_ZERO(v27)

      // Load the first 64 bytes of LHS and RHS data.
      "ld1 {v0.2d, v1.2d, v2.2d, v3.2d}, [%[lhs_ptr]], #64\n"
      "ld1 {v4.2d, v5.2d, v6.2d, v7.2d}, [%[rhs_ptr]], #64\n"

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for.
      // The RHS is stored in col-wise. Therefore, for 64-bit elements,
      // one register can hold 2 levels of depth.
      "mov w1, #2\n"

      // Main loop of the whole GEMM, over rows and columns of the
      // destination matrix.
      "1:\n"

      LCE_BMLA(v24, v4, v0, v1, v2, v3)

#if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
      "cmp w12, #8\n"
      "blt 78f\n"
      "and w2, w12, #-4\n"

      // Load the next 64 bytes of LHS and RHS data.
      "ld1 {v8.2d, v9.2d, v10.2d, v11.2d}, [%[lhs_ptr]], #64\n"
      "ld1 {v12.2d, v13.2d, v14.2d, v15.2d}, [%[rhs_ptr]], #64\n"

      "mov w1, #4\n"

      "80:\n"

      // loading v4
      "ld1 {v4.2d}, [%[rhs_ptr]], #16\n"

      LCE_BMLA_LD_RHS(v25, v5, v0, v1, v2, v3)
      LCE_BMLA_LD_RHS(v26, v6, v0, v1, v2, v3)
      LCE_BMLA_LD_ALL(v27, v7, v0, v1, v2, v3)
      LCE_BMLA_LD_RHS(v24, v12, v8, v9, v10, v11)
      LCE_BMLA_LD_RHS(v25, v13, v8, v9, v10, v11)
      LCE_BMLA_LD_RHS(v26, v14, v8, v9, v10, v11)
      LCE_BMLA_LD_ALL(v27, v15, v8, v9, v10, v11)

      LCE_BMLA(v24, v4, v0, v1, v2, v3)

      "add w1, w1, #4\n"
      "cmp w1, w2\n"
      "blt 80b\n"

      LCE_BMLA(v24, v12, v8, v9, v10, v11)
      LCE_BMLA(v25, v13, v8, v9, v10, v11)
      LCE_BMLA(v26, v14, v8, v9, v10, v11)
      LCE_BMLA(v27, v15, v8, v9, v10, v11)

      "78:\n"
#endif

      // Accumulation loop
      "cmp w1, w12\n"
      "beq 79f\n"

      "2:\n"

      // loading v4
      "ld1 {v4.2d}, [%[rhs_ptr]], #16\n"

      LCE_BMLA_LD_RHS(v25, v5, v0, v1, v2, v3)
      LCE_BMLA_LD_RHS(v26, v6, v0, v1, v2, v3)
      LCE_BMLA_LD_ALL(v27, v7, v0, v1, v2, v3)

      "add w1, w1, #2\n"
      "cmp w1, w12\n"

      LCE_BMLA(v24, v4, v0, v1, v2, v3)

      "blt 2b\n"

      "79:\n"

      LCE_BMLA(v25, v5, v0, v1, v2, v3)
      LCE_BMLA(v26, v6, v0, v1, v2, v3)
      LCE_BMLA(v27, v7, v0, v1, v2, v3)

      // End of accumulation. The registers v24 -- v27 contain the final
      // int32 accumulator values of the current 4x4 destination block.

      // Logic to advance to the next block in preparation for the next
      // iteration of the main loop. For now, we only want to compute
      // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
      // not yet ready to update the values of row and col, as we still need
      // the current values for the rest of the work on the current block.

      "cmp %w[row], w7\n"  // Have we finished the last row?
      "bge 4f\n"           // If finished last row, go to 4
      // Not finished last row: then advance to next row.
      // x9 is the LHS stride
      "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
      "b 5f\n"
      "4:\n"  // Finished last row...
      "mov %[lhs_col_ptr], x5\n"  // Go back to first row
      // Now we need to advance to the next column. If we already
      // finished the last column, then in principle we are done, however
      // we can't just return here, as we need to allow the end work of the
      // current block to complete. The good news is that at this point it
      // doesn't matter what data we load for the next column, since
      // we will exit from the main loop below before actually storing
      // anything computed from that data.
      "cmp %w[col], w8\n"  // Have we finished the last column?
      "bge 5f\n" // If yes, just carry on without updating the column pointer.
      // Not finished last column: then advance to next column.
      // x10 is the RHS stride
      "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
      "5:\n"

      // Set the LHS and RHS data pointers to the start of the columns just
      // computed.
      "mov %[lhs_ptr], %[lhs_col_ptr]\n"
      "mov %[rhs_ptr], %[rhs_col_ptr]\n"

      // Load some parameters needed for the end work on current block.
      "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"

      // Load backtransform add (duplicate 4 times into v13)
      "ldr w1, [%[params], #" RUY_STR(RUY_OFFSET_BACKTRANSFORM_ADD) "]\n"
      "dup v13.4s, w1 \n"

      // Load multiplication bias
      "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_POST_ACTIVATION_MULTIPLIER) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "add x2, x1, %x[row], lsl #2\n"
      "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
      "csel x1, x1, x2, eq\n"
      // Load 4 bias-multiplication values.
      "ld1 {v14.4s}, [x1], #16\n"

      // Load addition bias
      "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_POST_ACTIVATION_BIAS) "]\n"
      // Offset these base pointers as needed given the current row, col.
      "add x2, x1, %x[row], lsl #2\n"
      "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
      "csel x1, x1, x2, eq\n"
      // Load 4 bias-addition values.
      "ld1 {v15.4s}, [x1], #16\n"

      // Now that we know what LHS and RHS data the next iteration of the
      // main loop will need to load, we start loading the first 64 bytes of
      // each of LHS and RHS, into v0 -- v3 and v4 -- v7 as we don't need
      // them anymore in the rest of the work on the current block.
      "ld1 {v0.2d, v1.2d, v2.2d, v3.2d}, [%[lhs_ptr]], #64\n"
      "ld1 {v4.2d, v5.2d, v6.2d, v7.2d}, [%[rhs_ptr]], #64\n"

      // Perform the backtransformation (in int32)
      "shl v24.4s, v24.4s, #1\n"
      "shl v25.4s, v25.4s, #1\n"
      "shl v26.4s, v26.4s, #1\n"
      "shl v27.4s, v27.4s, #1\n"

      // Load the clamp_max bound (in parallel with the sub)
      "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
      "dup v12.4s, w2\n"  // clamp_min

      "sub v24.4s, v13.4s, v24.4s\n"
      "sub v25.4s, v13.4s, v25.4s\n"
      "sub v26.4s, v13.4s, v26.4s\n"
      "sub v27.4s, v13.4s, v27.4s\n"

      // Load the clamp_max bound (in parallel with the clamp_min)
      "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
      "dup v13.4s, w3\n"  // clamp_max

      // Perform the activation function, by clamping
      // Apply the clamp_min bound
      "smax v24.4s, v24.4s, v12.4s\n"
      "smax v25.4s, v25.4s, v12.4s\n"
      "smax v26.4s, v26.4s, v12.4s\n"
      "smax v27.4s, v27.4s, v12.4s\n"
      // Apply the clamp_max bound
      "smin v24.4s, v24.4s, v13.4s\n"
      "smin v25.4s, v25.4s, v13.4s\n"
      "smin v26.4s, v26.4s, v13.4s\n"
      "smin v27.4s, v27.4s, v13.4s\n"

      // Convert to single precision float
      "scvtf v24.4s, v24.4s\n"
      "scvtf v25.4s, v25.4s\n"
      "scvtf v26.4s, v26.4s\n"
      "scvtf v27.4s, v27.4s\n"

      // Perform the post multiplications
      "fmul v24.4s, v24.4s, v14.4s\n"
      "fmul v25.4s, v25.4s, v14.4s\n"
      "fmul v26.4s, v26.4s, v14.4s\n"
      "fmul v27.4s, v27.4s, v14.4s\n"
      // Perform the post additions
      "fadd v24.4s, v24.4s, v15.4s\n"
      "fadd v25.4s, v25.4s, v15.4s\n"
      "fadd v26.4s, v26.4s, v15.4s\n"
      "fadd v27.4s, v27.4s, v15.4s\n"

      // Compute how much of the 4x4 block of destination values that
      // we have computed, fit in the destination matrix. Typically, all of
      // it fits, but when the destination matrix shape is not a multiple
      // of 4x4, there are some 4x4 blocks along the boundaries that do
      // not fit entirely.
      "sub w1, %w[dst_rows], %w[row]\n"
      "sub w2, %w[dst_cols], %w[col]\n"
      "mov w3, #4\n"
      "cmp w1, #4\n"
      // Compute w1 = how many rows of the 4x4 block fit
      "csel w1, w1, w3, le\n"
      "cmp w2, #4\n"
      // Compute w2 = how many cols of the 4x4 block fit
      "csel w2, w2, w3, le\n"

      // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
      "cmp w1, w3\n"
      "ccmp w2, w3, 0, eq\n"
      // Yes, all of the 4x4 block fits, go to fast path.
      "beq 30f\n"
      // Not all of the 4x4 block fits.
      // Set (x3 address, x4 stride) to write to dst_tmp_buf
      "mov x3, %[dst_tmp_buf]\n"
      "mov x4, #16\n"
      "b 31f\n"
      "30:\n"
      // Yes, all of the 4x4 block fits.
      // Set (x3 address, x4 stride) to write directly to destination matrix.
      "mov x3, %[dst_ptr]\n"
      "mov x4, x11\n"
      "31:\n"

      // Write our values to the destination described by
      // (x3 address, x4 stride).
      "str q24, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v24)
      "str q25, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v25)
      "str q26, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v26)
      "str q27, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v27)

      // If all of the 4x4 block fits, we just finished writing it to the
      // destination, so we skip the next part.
      "beq 41f\n"
      // Not all of the 4x4 block fits in the destination matrix.  We just
      // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
      // it to copy into the destination matrix the part that fits.
      "mov x3, %[dst_tmp_buf]\n"
      "mov x4, %[dst_ptr]\n"
      "mov w14, #0\n"
      "50:\n"
      "mov w15, #0\n"
      "51:\n"
      "ldr w13, [x3, x15, lsl #2]\n"
      "str w13, [x4, x15, lsl #2]\n"
      "add w15, w15, #1\n"
      "cmp w15, w1\n"
      "blt 51b\n"
      "add w14, w14, #1\n"
      "add x3, x3, #16\n"
      "add x4, x4, x11\n"
      "cmp w14, w2\n"
      "blt 50b\n"
      "41:\n"
      "add %[dst_ptr], %[dst_ptr], #16\n"

      // At this point we have completely finished writing values to the
      // destination matrix for the current block.

      // Move to the next block of the destination matrix, for the next iter
      // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
      // been updated earlier.
      // Have we reached the end row?
      "cmp %w[row], w7\n"
      "beq 20f\n"  // yes, end row.
      // Not end row. Move to the next row.
      "add %w[row], %w[row], #4\n"
      "b 21f\n"
      "20:\n"
      // Was already at end row.
      "mov %w[row], w6\n"  // Move back to first row.
      "add %w[col], %w[col], #4\n"  // Move to the next column.
      "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
      "mov %[dst_ptr], %[dst_col_ptr]\n"
      "21:\n"

      // Main loop exit condition: have we hit the end column?
      "cmp %w[col], w8\n"

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for.
      "mov w1, #2\n"

      "ble 1b\n"

      // clang-format on

      : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
      : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
        [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
      : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "cc",
        "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
}

#undef RUY_OFFSET_BACKTRANSFORM_ADD
#undef RUY_OFFSET_POST_ACTIVATION_MULTIPLIER
#undef RUY_OFFSET_POST_ACTIVATION_BIAS
#undef RUY_OFFSET_FLAGS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR

#endif  // RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
