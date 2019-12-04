#include <cstdint>

#include "bgemm_kernels_common.h"
#include "profiling/instrumentation.h"

using namespace ruy;

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 8
#define RUY_OFFSET_DST_BASE_PTR 16
#define RUY_OFFSET_BIAS 24
#define RUY_OFFSET_START_ROW 32
#define RUY_OFFSET_START_COL 36
#define RUY_OFFSET_LAST_ROW 40
#define RUY_OFFSET_LAST_COL 44
#define RUY_OFFSET_LHS_STRIDE 56
#define RUY_OFFSET_RHS_STRIDE 60
#define RUY_OFFSET_DST_STRIDE 64
#define RUY_OFFSET_DEPTH 68
#define RUY_OFFSET_CLAMP_MIN 72
#define RUY_OFFSET_CLAMP_MAX 76
#define RUY_OFFSET_FLAGS 80

template <typename Params>
void CheckOffsetsInKernelParams32BP(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
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
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

void BinaryKernelNeonOutOfOrder32BP4x4(
    const BinaryKernelParams32BP<4, 4>& params) {
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

  // The asm kernel below has the following NEON register allocation:
  // kernel layout TODO:

  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

#define LCE_BMLA_BP32(Vd, Vr, Vl1, Vl2, Vl3, Vl4) \
  "eor v26.16b, " #Vr ".16b, " #Vl1               \
  ".16b         \n"                               \
  "eor v27.16b, " #Vr ".16b, " #Vl2               \
  ".16b         \n"                               \
  "eor v28.16b, " #Vr ".16b, " #Vl3               \
  ".16b         \n"                               \
  "eor v29.16b, " #Vr ".16b, " #Vl4               \
  ".16b         \n"                               \
  "cnt v26.16b, v26.16b                 \n"       \
  "cnt v27.16b, v27.16b                 \n"       \
  "cnt v28.16b, v28.16b                 \n"       \
  "cnt v29.16b, v29.16b                 \n"       \
  "addv b26, v26.16b                \n"           \
  "addv b27, v27.16b                \n"           \
  "addv b28, v28.16b                \n"           \
  "addv b29, v29.16b                \n"           \
  "ins v26.s[1], v27.s[0]             \n"         \
  "ins v26.s[2], v28.s[0]             \n"         \
  "ins v26.s[3], v29.s[0]             \n"         \
  "add " #Vd ".4s, " #Vd ".4s, v26.4s           \n"

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
      "ld1 {v18.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v19.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v20.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v21.4s}, [%[lhs_ptr]], #16\n"

      "ld1 {v12.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v13.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"

      // Clear accumulators.
      RUY_MAKE_ZERO(v0)
      RUY_MAKE_ZERO(v2)
      RUY_MAKE_ZERO(v4)
      RUY_MAKE_ZERO(v6)

      // w1 is the number of levels of depth that we have already loaded
      // LHS and RHS data for. Corresponding to the initial ld1 instructions
      // above, this is currently 1.
      // "mov w1, #1\n"
      // the RHS is stored in col-wise. For 32-bit elements,
      // one register can hold 4 levels of depth.
      "mov w1, #4\n"

      // Main loop of the whole GEMM, over rows and columns of the
      // destination matrix.
      "1:\n"

      LCE_BMLA_BP32(v0, v12, v18, v19, v20, v21)
      LCE_BMLA_BP32(v2, v13, v18, v19, v20, v21)
      LCE_BMLA_BP32(v4, v14, v18, v19, v20, v21)
      LCE_BMLA_BP32(v6, v15, v18, v19, v20, v21)

      // Accumulation loop
      "cmp w1, w12\n"
      "beq 79f\n"

      "2:\n"
      "ld1 {v18.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v19.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v20.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v21.4s}, [%[lhs_ptr]], #16\n"

      "ld1 {v12.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v13.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"

      "add w1, w1, #4\n"
      "cmp w1, w12\n"

      LCE_BMLA_BP32(v0, v12, v18, v19, v20, v21)
      LCE_BMLA_BP32(v2, v13, v18, v19, v20, v21)
      LCE_BMLA_BP32(v4, v14, v18, v19, v20, v21)
      LCE_BMLA_BP32(v6, v15, v18, v19, v20, v21)

      "blt 2b\n"

      "79:\n"

      // End of accumulation. The registers v0 -- v11 contain the final
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
      "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

      // Offset these base pointers as needed given the current row, col.
      "add x5, x1, %x[row], lsl #2\n"

      "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
      "csel x1, x1, x5, eq\n"

      // Now that we know what LHS and RHS data the next iteration of the
      // main loop will need to load, we start loading the first 64 bytes of
      // each of LHS and RHS, into v18 -- v21 and v12 -- v15 as we don't need
      // them anymore in the rest of the work on the current block.
      "ld1 {v18.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v19.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v20.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v21.4s}, [%[lhs_ptr]], #16\n"

      "ld1 {v12.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v13.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"

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

      // convert to single precision float before storing the NEON registers
      "ucvtf v0.4s, v0.4s\n"
      "ucvtf v2.4s, v2.4s\n"
      "ucvtf v4.4s, v4.4s\n"
      "ucvtf v6.4s, v6.4s\n"

      // Write our values to the destination described by
      // (x3 address, x4 stride).
      "str q0, [x3, #0]\n"
      "add x3, x3, x4\n"
      "str q2, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v0)
      RUY_MAKE_ZERO(v2)
      "str q4, [x3, #0]\n"
      "add x3, x3, x4\n"
      "str q6, [x3, #0]\n"
      "add x3, x3, x4\n"
      RUY_MAKE_ZERO(v4)
      RUY_MAKE_ZERO(v6)

      // If all of the 4x4 block fits, we just finished writing it to the
      // destination, so we skip the next part.
      "beq 41f\n"
      // Not all of the 4x4 block fits in the destination matrix.  We just
      // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
      // it to copy into the destination matrix the part that fits.
      "mov x3, %[dst_tmp_buf]\n"
      "mov x4, %[dst_ptr]\n"
      "mov w6, #0\n"
      "50:\n"
      "mov w5, #0\n"
      "51:\n"
      "ldr w7, [x3, x5, lsl #2]\n"
      "str w7, [x4, x5, lsl #2]\n"
      "add w5, w5, #1\n"
      "cmp w5, w1\n"
      "blt 51b\n"
      "add w6, w6, #1\n"
      "add x3, x3, #16\n"
      "add x4, x4, x11\n"
      "cmp w6, w2\n"
      "blt 50b\n"
      "41:\n"
      "add %[dst_ptr], %[dst_ptr], #16\n"

      // At this point we have completely finished writing values to the
      // destination matrix for the current block.

      // Reload some params --- we had used x5 -- x7 for a few other things
      // since the last time we had loaded them.
      "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
      "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
      "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

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
      : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
        "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
}

#undef RUY_OFFSET_BIAS
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
