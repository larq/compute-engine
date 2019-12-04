#include <bitset>
#include <cstdint>

#include "bgemm_kernels_common.h"
#include "profiling/instrumentation.h"
// #include "tensorflow/lite/experimental/ruy/common.h"
// #include "tensorflow/lite/experimental/ruy/kernel.h"
// #include "tensorflow/lite/experimental/ruy/opt_set.h"
// #include "tensorflow/lite/experimental/ruy/platform.h"

// #include <sstream>

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

// TODO: this needs to be double checked if the params offset are still valid
// for different types
template <typename Params>
void CheckOffsetsInKernelParamsFloat(const Params&) {
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

// Just a plain float kernel; good enough for out-of-order cores.
// The closest to it in the gemmlowp collection would be
// NEON_64bit_GEMM_Float32_WithScalar,
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L3925
//
// Besides ruy-ification, the main nuance here is that we stick to a 8x8
// width instead of the wider 12x8 that the register space permits and that
// the aforementioned gemmlowp kernel uses.  Ruy likes powers of two for now
// and we don't have evidence that going beyond 8x8 is needed.
void BinaryKernelNeonOutOfOrder32(const BinaryKernelParams32<8, 8>& params) {
  CheckOffsetsInKernelParamsFloat(params);
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  // const float* lhs_col_ptr = params.lhs_base_ptr;
  // const float* rhs_col_ptr = params.rhs_base_ptr;
  // const float* lhs_ptr = lhs_col_ptr;
  // const float* rhs_ptr = rhs_col_ptr;
  // const std::uint32_t* lhs_col_ptr = params.lhs_base_ptr;
  // const std::uint32_t* rhs_col_ptr = params.rhs_base_ptr;
  // const std::uint32_t* lhs_ptr = lhs_col_ptr;
  // const std::uint32_t* rhs_ptr = rhs_col_ptr;
  std::uint32_t* lhs_col_ptr = const_cast<std::uint32_t*>(params.lhs_base_ptr);
  std::uint32_t* rhs_col_ptr = const_cast<std::uint32_t*>(params.rhs_base_ptr);
  std::uint32_t* lhs_ptr = lhs_col_ptr;
  std::uint32_t* rhs_ptr = rhs_col_ptr;

  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // std::cout << "Binary Kernel" << std::endl;

  // lhs_ptr[0] = 0xEFEFEFEF;
  // rhs_ptr[0] = 0xFEFEFEFE;

  // lhs_ptr[1] = 0xEFEFEFEF;
  // rhs_ptr[1] = 0xFEFEFEFE;

  // lhs_ptr[2] = 0xEFEFEFEF;
  // rhs_ptr[2] = 0xFEFEFEFE;

  // lhs_ptr[3] = 0xEFEFEFEF;
  // rhs_ptr[3] = 0xFEFEFEFE;

  // auto print_bits = [](std::uint32_t x) {
  //   std::string a;
  //   for (int i = 31; i >= 0; --i) {
  //     a += (x & (1 << i)) ? "1" : "0";
  //     if (i % 8 == 0) a += " ";
  //   }
  //   return a;
  // };

  // std::cout << "V0.s[0] = " << lhs_ptr[0] << "->" << print_bits(lhs_ptr[0])
  //           << std::endl;
  // std::cout << "V0.s[1] = " << lhs_ptr[1] << "->" << print_bits(lhs_ptr[1])
  //           << std::endl;
  // std::cout << "V0.s[2] = " << lhs_ptr[2] << "->" << print_bits(lhs_ptr[2])
  //           << std::endl;
  // std::cout << "V0.s[3] = " << lhs_ptr[3] << "->" << print_bits(lhs_ptr[3])
  //           << std::endl;

  // std::cout << "V0.s[0] = " << lhs_ptr[0] << "->" <<
  // std::bitset<32>(lhs_ptr[0]) << std::endl; std::cout << "V0.s[1] = " <<
  // lhs_ptr[1] << "->" << std::bitset<32>(lhs_ptr[1]) << std::endl; std::cout
  // << "V0.s[2] = " << lhs_ptr[2] << "->" << std::bitset<32>(lhs_ptr[2]) <<
  // std::endl; std::cout << "V0.s[3] = " << lhs_ptr[3] << "->" <<
  // std::bitset<32>(lhs_ptr[3]) << std::endl;

  // std::cout << "V1.s[0] = " << lhs_ptr[4] << "->" <<
  // std::bitset<32>(lhs_ptr[4]) << std::endl; std::cout << "V1.s[1] = " <<
  // lhs_ptr[5] << "->" << std::bitset<32>(lhs_ptr[5]) << std::endl; std::cout
  // << "V1.s[2] = " << lhs_ptr[6] << "->" << std::bitset<32>(lhs_ptr[6]) <<
  // std::endl; std::cout << "V1.s[3] = " << lhs_ptr[7] << "->" <<
  // std::bitset<32>(lhs_ptr[7]) << std::endl;

  // std::cout << "V2.s[0] = " << rhs_ptr[0] << "->" << print_bits(rhs_ptr[0])
  //           << std::endl;

  // std::cout << "V2.s[0] = " << rhs_ptr[0] << "->" <<
  // std::bitset<32>(rhs_ptr[0]) << std::endl; std::cout << "V2.s[1] = " <<
  // rhs_ptr[1] << "->" << std::bitset<32>(rhs_ptr[1]) << std::endl; std::cout
  // << "V2.s[2] = " << rhs_ptr[2] << "->" << std::bitset<32>(rhs_ptr[2]) <<
  // std::endl; std::cout << "V2.s[3] = " << rhs_ptr[3] << "->" <<
  // std::bitset<32>(rhs_ptr[3]) << std::endl;

  // std::cout << "V3.s[0] = " << rhs_ptr[4] << "->" <<
  // std::bitset<32>(rhs_ptr[4]) << std::endl; std::cout << "V3.s[1] = " <<
  // rhs_ptr[5] << "->" << std::bitset<32>(rhs_ptr[5]) << std::endl; std::cout
  // << "V3.s[2] = " << rhs_ptr[6] << "->" << std::bitset<32>(rhs_ptr[6]) <<
  // std::endl; std::cout << "V3.s[3] = " << rhs_ptr[7] << "->" <<
  // std::bitset<32>(rhs_ptr[7]) << std::endl;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v15 are used to load data from LHS and RHS.
  // At least v0 and v1 are used to load a 8x1 block of LHS, and v2 and
  // v3 are used to load a 1x8 block of RHS, like this:
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------\
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------\
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for floading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

#define LCE_BMLA_BP32(Vd, Vn, Vm)\
      "dup v6.4s, " #Vm "\n"\
      "eor v5.16b, " #Vn".16b , v6.16b\n"\
        "cnt v5.16b, v5.16b\n"\
        "addp v5.16b, v5.16b, v5.16b\n"\
        "addp v5.16b, v5.16b, v5.16b\n"\
        RUY_MAKE_ZERO(v6)\
        "mov v6.b[0], v5.b[0]\n"\
        "mov v6.b[4], v5.b[1]\n"\
        "mov v6.b[8], v5.b[2]\n"\
        "mov v6.b[12], v5.b[3]\n"\
        "add "#Vd".4s, "#Vd".4s, v6.4s\n"\

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

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // "fmla v16.4s, v0.4s, v2.s[0]\n"
        LCE_BMLA_BP32(v16, v0, v2.s[0])

        // "fmla v18.4s, v0.4s, v2.s[1]\n"
        LCE_BMLA_BP32(v18, v0, v2.s[1])

        // "fmla v20.4s, v0.4s, v2.s[2]\n"
        LCE_BMLA_BP32(v20, v0, v2.s[2])

        // "fmla v22.4s, v0.4s, v2.s[3]\n"
        LCE_BMLA_BP32(v22, v0, v2.s[3])

// #if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
//         "cmp w12, #8\n"
//         "blt 78f\n"
//         "and w2, w12, #-4\n"

//         "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v5.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v6.4s}, [%[rhs_ptr]], #16\n"
//         "ld1 {v7.4s}, [%[rhs_ptr]], #16\n"

//         "ld1 {v8.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v9.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v10.4s}, [%[rhs_ptr]], #16\n"
//         "ld1 {v11.4s}, [%[rhs_ptr]], #16\n"

//         "ld1 {v12.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v13.4s}, [%[lhs_ptr]], #16\n"
//         "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
//         "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"
//         "mov w1, #4\n"

//         "80:\n"

//         "add %[lhs_ptr], %[lhs_ptr], #128\n"
//         "add %[rhs_ptr], %[rhs_ptr], #128\n"

//         "fmla v24.4s, v0.4s, v3.s[0]\n"
//         "fmla v26.4s, v0.4s, v3.s[1]\n"
//         "fmla v28.4s, v0.4s, v3.s[2]\n"
//         "fmla v30.4s, v0.4s, v3.s[3]\n"
//         "ldr q0, [%[lhs_ptr], #-128]\n"
//         "fmla v25.4s, v1.4s, v3.s[0]\n"
//         "fmla v27.4s, v1.4s, v3.s[1]\n"
//         "fmla v29.4s, v1.4s, v3.s[2]\n"
//         "fmla v31.4s, v1.4s, v3.s[3]\n"
//         "ldr q3, [%[rhs_ptr], #-112]\n"
//         "fmla v17.4s, v1.4s, v2.s[0]\n"
//         "fmla v19.4s, v1.4s, v2.s[1]\n"
//         "fmla v21.4s, v1.4s, v2.s[2]\n"
//         "fmla v23.4s, v1.4s, v2.s[3]\n"
//         "ldr q1, [%[lhs_ptr], #-112]\n"
//         "fmla v16.4s, v4.4s, v6.s[0]\n"
//         "fmla v18.4s, v4.4s, v6.s[1]\n"
//         "ldr q2, [%[rhs_ptr], #-128]\n"
//         "fmla v20.4s, v4.4s, v6.s[2]\n"
//         "fmla v22.4s, v4.4s, v6.s[3]\n"

//         "fmla v24.4s, v4.4s, v7.s[0]\n"
//         "fmla v26.4s, v4.4s, v7.s[1]\n"
//         "fmla v28.4s, v4.4s, v7.s[2]\n"
//         "fmla v30.4s, v4.4s, v7.s[3]\n"
//         "ldr q4, [%[lhs_ptr], #-96]\n"
//         "fmla v25.4s, v5.4s, v7.s[0]\n"
//         "fmla v27.4s, v5.4s, v7.s[1]\n"
//         "fmla v29.4s, v5.4s, v7.s[2]\n"
//         "fmla v31.4s, v5.4s, v7.s[3]\n"
//         "ldr q7, [%[rhs_ptr], #-80]\n"
//         "fmla v17.4s, v5.4s, v6.s[0]\n"
//         "fmla v19.4s, v5.4s, v6.s[1]\n"
//         "fmla v21.4s, v5.4s, v6.s[2]\n"
//         "fmla v23.4s, v5.4s, v6.s[3]\n"
//         "ldr q5, [%[lhs_ptr], #-80]\n"
//         "fmla v16.4s, v8.4s, v10.s[0]\n"
//         "fmla v18.4s, v8.4s, v10.s[1]\n"
//         "ldr q6, [%[rhs_ptr], #-96]\n"
//         "fmla v20.4s, v8.4s, v10.s[2]\n"
//         "fmla v22.4s, v8.4s, v10.s[3]\n"

//         "fmla v24.4s, v8.4s, v11.s[0]\n"
//         "fmla v26.4s, v8.4s, v11.s[1]\n"
//         "fmla v28.4s, v8.4s, v11.s[2]\n"
//         "fmla v30.4s, v8.4s, v11.s[3]\n"
//         "ldr q8, [%[lhs_ptr], #-64]\n"
//         "fmla v25.4s, v9.4s, v11.s[0]\n"
//         "fmla v27.4s, v9.4s, v11.s[1]\n"
//         "fmla v29.4s, v9.4s, v11.s[2]\n"
//         "fmla v31.4s, v9.4s, v11.s[3]\n"
//         "ldr q11, [%[rhs_ptr], #-48]\n"
//         "fmla v17.4s, v9.4s, v10.s[0]\n"
//         "fmla v19.4s, v9.4s, v10.s[1]\n"
//         "fmla v21.4s, v9.4s, v10.s[2]\n"
//         "fmla v23.4s, v9.4s, v10.s[3]\n"
//         "ldr q9, [%[lhs_ptr], #-48]\n"
//         "fmla v16.4s, v12.4s, v14.s[0]\n"
//         "fmla v18.4s, v12.4s, v14.s[1]\n"
//         "ldr q10, [%[rhs_ptr], #-64]\n"
//         "fmla v20.4s, v12.4s, v14.s[2]\n"
//         "fmla v22.4s, v12.4s, v14.s[3]\n"

//         "fmla v24.4s, v12.4s, v15.s[0]\n"
//         "fmla v26.4s, v12.4s, v15.s[1]\n"
//         "fmla v28.4s, v12.4s, v15.s[2]\n"
//         "fmla v30.4s, v12.4s, v15.s[3]\n"
//         "ldr q12, [%[lhs_ptr], #-32]\n"
//         "fmla v25.4s, v13.4s, v15.s[0]\n"
//         "fmla v27.4s, v13.4s, v15.s[1]\n"
//         "fmla v29.4s, v13.4s, v15.s[2]\n"
//         "fmla v31.4s, v13.4s, v15.s[3]\n"
//         "ldr q15, [%[rhs_ptr], #-16]\n"
//         "fmla v17.4s, v13.4s, v14.s[0]\n"
//         "fmla v19.4s, v13.4s, v14.s[1]\n"
//         "fmla v21.4s, v13.4s, v14.s[2]\n"
//         "fmla v23.4s, v13.4s, v14.s[3]\n"
//         "ldr q13, [%[lhs_ptr], #-16]\n"
//         "fmla v16.4s, v0.4s, v2.s[0]\n"
//         "fmla v18.4s, v0.4s, v2.s[1]\n"
//         "ldr q14, [%[rhs_ptr], #-32]\n"
//         "fmla v20.4s, v0.4s, v2.s[2]\n"
//         "fmla v22.4s, v0.4s, v2.s[3]\n"

//         "add w1, w1, #4\n"
//         "cmp w1, w2\n"
//         "blt 80b\n"

//         "fmla v16.4s, v4.4s, v6.s[0]\n"
//         "fmla v18.4s, v4.4s, v6.s[1]\n"
//         "fmla v20.4s, v4.4s, v6.s[2]\n"
//         "fmla v22.4s, v4.4s, v6.s[3]\n"
//         "fmla v24.4s, v4.4s, v7.s[0]\n"
//         "fmla v26.4s, v4.4s, v7.s[1]\n"
//         "fmla v28.4s, v4.4s, v7.s[2]\n"
//         "fmla v30.4s, v4.4s, v7.s[3]\n"
//         "fmla v25.4s, v5.4s, v7.s[0]\n"
//         "fmla v27.4s, v5.4s, v7.s[1]\n"
//         "fmla v29.4s, v5.4s, v7.s[2]\n"
//         "fmla v31.4s, v5.4s, v7.s[3]\n"
//         "fmla v17.4s, v5.4s, v6.s[0]\n"
//         "fmla v19.4s, v5.4s, v6.s[1]\n"
//         "fmla v21.4s, v5.4s, v6.s[2]\n"
//         "fmla v23.4s, v5.4s, v6.s[3]\n"

//         "fmla v16.4s, v8.4s, v10.s[0]\n"
//         "fmla v18.4s, v8.4s, v10.s[1]\n"
//         "fmla v20.4s, v8.4s, v10.s[2]\n"
//         "fmla v22.4s, v8.4s, v10.s[3]\n"
//         "fmla v24.4s, v8.4s, v11.s[0]\n"
//         "fmla v26.4s, v8.4s, v11.s[1]\n"
//         "fmla v28.4s, v8.4s, v11.s[2]\n"
//         "fmla v30.4s, v8.4s, v11.s[3]\n"
//         "fmla v25.4s, v9.4s, v11.s[0]\n"
//         "fmla v27.4s, v9.4s, v11.s[1]\n"
//         "fmla v29.4s, v9.4s, v11.s[2]\n"
//         "fmla v31.4s, v9.4s, v11.s[3]\n"
//         "fmla v17.4s, v9.4s, v10.s[0]\n"
//         "fmla v19.4s, v9.4s, v10.s[1]\n"
//         "fmla v21.4s, v9.4s, v10.s[2]\n"
//         "fmla v23.4s, v9.4s, v10.s[3]\n"

//         "fmla v16.4s, v12.4s, v14.s[0]\n"
//         "fmla v18.4s, v12.4s, v14.s[1]\n"
//         "fmla v20.4s, v12.4s, v14.s[2]\n"
//         "fmla v22.4s, v12.4s, v14.s[3]\n"
//         "fmla v24.4s, v12.4s, v15.s[0]\n"
//         "fmla v26.4s, v12.4s, v15.s[1]\n"
//         "fmla v28.4s, v12.4s, v15.s[2]\n"
//         "fmla v30.4s, v12.4s, v15.s[3]\n"
//         "fmla v25.4s, v13.4s, v15.s[0]\n"
//         "fmla v27.4s, v13.4s, v15.s[1]\n"
//         "fmla v29.4s, v13.4s, v15.s[2]\n"
//         "fmla v31.4s, v13.4s, v15.s[3]\n"
//         "fmla v17.4s, v13.4s, v14.s[0]\n"
//         "fmla v19.4s, v13.4s, v14.s[1]\n"
//         "fmla v21.4s, v13.4s, v14.s[2]\n"
//         "fmla v23.4s, v13.4s, v14.s[3]\n"

//         "78:\n"
// #endif

        // Accumulation loop
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"
        // "fmla v24.4s, v0.4s, v3.s[0]\n"
        // "fmla v26.4s, v0.4s, v3.s[1]\n"
        LCE_BMLA_BP32(v24, v0, v3.s[0])
        LCE_BMLA_BP32(v26, v0, v3.s[1])

        "ld1 {v4.4s}, [%[rhs_ptr]], #16\n"

        // "fmla v28.4s, v0.4s, v3.s[2]\n"
        // "fmla v30.4s, v0.4s, v3.s[3]\n"
        LCE_BMLA_BP32(v28, v0, v3.s[2])
        LCE_BMLA_BP32(v30, v0, v3.s[3])

        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"

        // "fmla v25.4s, v1.4s, v3.s[0]\n"
        // "fmla v27.4s, v1.4s, v3.s[1]\n"
        LCE_BMLA_BP32(v25, v1, v3.s[0])
        LCE_BMLA_BP32(v27, v1, v3.s[1])

        "add w1, w1, #1\n"

        // "fmla v29.4s, v1.4s, v3.s[2]\n"
        // "fmla v31.4s, v1.4s, v3.s[3]\n"
        LCE_BMLA_BP32(v29, v1, v3.s[2])
        LCE_BMLA_BP32(v31, v1, v3.s[3])

        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // "fmla v17.4s, v1.4s, v2.s[0]\n"
        // "fmla v19.4s, v1.4s, v2.s[1]\n"
        LCE_BMLA_BP32(v17, v1, v2.s[0])
        LCE_BMLA_BP32(v19, v1, v2.s[1])

        "cmp w1, w12\n"

        // "fmla v21.4s, v1.4s, v2.s[2]\n"
        // "fmla v23.4s, v1.4s, v2.s[3]\n"
        LCE_BMLA_BP32(v21, v1, v2.s[2])
        LCE_BMLA_BP32(v23, v1, v2.s[3])

        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"

        // "fmla v16.4s, v0.4s, v4.s[0]\n"
        // "fmla v18.4s, v0.4s, v4.s[1]\n"
        LCE_BMLA_BP32(v16, v0, v4.s[0])
        LCE_BMLA_BP32(v18, v0, v4.s[1])

        "mov v2.16b, v4.16b\n"

        // "fmla v20.4s, v0.4s, v4.s[2]\n"
        // "fmla v22.4s, v0.4s, v4.s[3]\n"
        LCE_BMLA_BP32(v20, v0, v4.s[2])
        LCE_BMLA_BP32(v22, v0, v4.s[3])

        "blt 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        // "fmla v24.4s, v0.4s, v3.s[0]\n"
        LCE_BMLA_BP32(v24, v0, v3.s[0])

        // "fmla v26.4s, v0.4s, v3.s[1]\n"
        LCE_BMLA_BP32(v26, v0, v3.s[1])

        // "fmla v28.4s, v0.4s, v3.s[2]\n"
        LCE_BMLA_BP32(v28, v0, v3.s[2])

        // "fmla v30.4s, v0.4s, v3.s[3]\n"
        LCE_BMLA_BP32(v30, v0, v3.s[3])

        // "fmla v25.4s, v1.4s, v3.s[0]\n"
        LCE_BMLA_BP32(v25, v1, v3.s[0])

        // "fmla v27.4s, v1.4s, v3.s[1]\n"
        LCE_BMLA_BP32(v27, v1, v3.s[1])

        // "fmla v29.4s, v1.4s, v3.s[2]\n"
        LCE_BMLA_BP32(v29, v1, v3.s[2])

        // "fmla v31.4s, v1.4s, v3.s[3]\n"
        LCE_BMLA_BP32(v31, v1, v3.s[3])

        // "fmla v17.4s, v1.4s, v2.s[0]\n"
        LCE_BMLA_BP32(v17, v1, v2.s[0])

        // "fmla v19.4s, v1.4s, v2.s[1]\n"
        LCE_BMLA_BP32(v19, v1, v2.s[1])

        // "fmla v21.4s, v1.4s, v2.s[2]\n"
        LCE_BMLA_BP32(v21, v1, v2.s[2])

        // "fmla v23.4s, v1.4s, v2.s[3]\n"
        LCE_BMLA_BP32(v23, v1, v2.s[3])

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
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
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
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

        // // Load 8 bias values.
        // "ld1 {v14.4s}, [x1], #16\n"
        // "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // // Perform the bias-addition (per the above, we have just folded into
        // // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // "fadd v16.4s, v16.4s, v14.4s\n"
        // "fadd v17.4s, v17.4s, v15.4s\n"
        // "fadd v18.4s, v18.4s, v14.4s\n"
        // "fadd v19.4s, v19.4s, v15.4s\n"
        // "fadd v20.4s, v20.4s, v14.4s\n"
        // "fadd v21.4s, v21.4s, v15.4s\n"
        // "fadd v22.4s, v22.4s, v14.4s\n"
        // "fadd v23.4s, v23.4s, v15.4s\n"
        // "fadd v24.4s, v24.4s, v14.4s\n"
        // "fadd v25.4s, v25.4s, v15.4s\n"
        // "fadd v26.4s, v26.4s, v14.4s\n"
        // "fadd v27.4s, v27.4s, v15.4s\n"
        // "fadd v28.4s, v28.4s, v14.4s\n"
        // "fadd v29.4s, v29.4s, v15.4s\n"
        // "fadd v30.4s, v30.4s, v14.4s\n"
        // "fadd v31.4s, v31.4s, v15.4s\n"

        // // Load the clamp_min, clamp_max bounds
        // "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        // "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        // "dup v14.4s, w2\n"  // clamp_min
        // "dup v15.4s, w3\n"  // clamp_max

        // // Apply the clamp_min bound
        // "fmax v16.4s, v16.4s, v14.4s\n"
        // "fmax v17.4s, v17.4s, v14.4s\n"
        // "fmax v18.4s, v18.4s, v14.4s\n"
        // "fmax v19.4s, v19.4s, v14.4s\n"
        // "fmax v20.4s, v20.4s, v14.4s\n"
        // "fmax v21.4s, v21.4s, v14.4s\n"
        // "fmax v22.4s, v22.4s, v14.4s\n"
        // "fmax v23.4s, v23.4s, v14.4s\n"
        // "fmax v24.4s, v24.4s, v14.4s\n"
        // "fmax v25.4s, v25.4s, v14.4s\n"
        // "fmax v26.4s, v26.4s, v14.4s\n"
        // "fmax v27.4s, v27.4s, v14.4s\n"
        // "fmax v28.4s, v28.4s, v14.4s\n"
        // "fmax v29.4s, v29.4s, v14.4s\n"
        // "fmax v30.4s, v30.4s, v14.4s\n"
        // "fmax v31.4s, v31.4s, v14.4s\n"

        // // Apply the clamp_max bound
        // "fmin v16.4s, v16.4s, v15.4s\n"
        // "fmin v17.4s, v17.4s, v15.4s\n"
        // "fmin v18.4s, v18.4s, v15.4s\n"
        // "fmin v19.4s, v19.4s, v15.4s\n"
        // "fmin v20.4s, v20.4s, v15.4s\n"
        // "fmin v21.4s, v21.4s, v15.4s\n"
        // "fmin v22.4s, v22.4s, v15.4s\n"
        // "fmin v23.4s, v23.4s, v15.4s\n"
        // "fmin v24.4s, v24.4s, v15.4s\n"
        // "fmin v25.4s, v25.4s, v15.4s\n"
        // "fmin v26.4s, v26.4s, v15.4s\n"
        // "fmin v27.4s, v27.4s, v15.4s\n"
        // "fmin v28.4s, v28.4s, v15.4s\n"
        // "fmin v29.4s, v29.4s, v15.4s\n"
        // "fmin v30.4s, v30.4s, v15.4s\n"
        // "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // convert to single precision float before storing the NEON registers
        "ucvtf v16.4s, v16.4s\n"
        "ucvtf v17.4s, v17.4s\n"
        "ucvtf v18.4s, v18.4s\n"
        "ucvtf v19.4s, v19.4s\n"

        "ucvtf v20.4s, v20.4s\n"
        "ucvtf v21.4s, v21.4s\n"
        "ucvtf v22.4s, v22.4s\n"
        "ucvtf v23.4s, v23.4s\n"

        "ucvtf v24.4s, v24.4s\n"
        "ucvtf v25.4s, v25.4s\n"
        "ucvtf v26.4s, v26.4s\n"
        "ucvtf v27.4s, v27.4s\n"

        "ucvtf v28.4s, v28.4s\n"
        "ucvtf v29.4s, v29.4s\n"
        "ucvtf v30.4s, v30.4s\n"
        "ucvtf v31.4s, v31.4s\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
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
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
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
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

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

  // std::cout << "DST_TMP[0] " << params.dst_tmp_buf[0] << "->"
  //           << print_bits(params.dst_tmp_buf[0]) << std::endl;
  // std::cout << "DST_TMP[1] " << params.dst_tmp_buf[1] << "->"
  //           << print_bits(params.dst_tmp_buf[1]) << std::endl;
  // std::cout << "DST_TMP[2] " << params.dst_tmp_buf[2] << "->"
  //           << print_bits(params.dst_tmp_buf[2]) << std::endl;
  // std::cout << "DST_TMP[3] " << params.dst_tmp_buf[3] << "->"
  //           << print_bits(params.dst_tmp_buf[3]) << std::endl;
  // std::cout << "DST[1] " << dst_ptr[0] << std::endl;
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
