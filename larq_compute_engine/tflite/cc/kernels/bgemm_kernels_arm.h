#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_

#include "bgemm_kernels_common.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/kernel_common.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/tune.h"


using namespace ruy;

#if RUY_PLATFORM(NEON) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// void BinaryKernelNeonOutOfOrder32(const BinaryKernelParams32<8, 8>& params);

#if RUY_PLATFORM(NEON_64)
// A BGEMM kernel for ARM64 Neon.

#include "bgemm_kernels_arm64.h"

// specialized kernel for 32-bit bitpacking, float output and 32-bit accumulator
template <>
struct BgemmKernel<ruy::Path::kNeon, std::uint32_t, std::uint32_t, float,
                   BasicSpec<std::int32_t, float>> {
  Tuning tuning = Tuning::kAuto;
  // TODO: whats the best kernel layout for ARM64 and int{8|32|64} data types?
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<std::uint32_t>& lhs,
           const ruy::PackedMatrix<std::uint32_t>& rhs,
           const BasicSpec<std::int32_t, float>& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<float>* dst) const {
    // using AccumScalar = float;// typename Spec::AccumScalar;
    // const int depth = lhs.layout.rows;
    // std::cout << "RUY Path: kNeon64" << std::endl;

    BinaryKernelParams32<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeBinaryKernelParams32(lhs, rhs, spec, start_row, start_col, end_row,
                             end_col, dst, &params);
    BinaryKernelNeonOutOfOrder32(params);

    // TODO: currently we ignore the tuning optimization
    // if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
    //   KernelFloatNeonInOrder(params);
    // } else {
    //   KernelFloatNeonOutOfOrder(params);
    // }

    // constexpr int RhsCells = 2;
    // float32x4_t acc[3][4 * RhsCells];
    // AccumScalar accum_ptr[3*4*RhsCells] = {0};

    // for (int i = 0; i < 3; i++) {
    //   for (int j = 0; j < 4 * RhsCells; j++) {
    //     acc[i][j] = vld1q_f32(accum_ptr + 4 * (i + 3 * j));
    //   }
    // }

    // for (int d = 0; d < depth; d++) {
    //   float32x4_t lhs[3];
    //   for (int i = 0; i < 3; i++) {
    //     lhs[i] = vld1q_f32(lhs_ptr + 4 * i);
    //   }
    //   float32x4_t rhs[RhsCells];
    //   for (int i = 0; i < RhsCells; i++) {
    //     rhs[i] = vld1q_f32(rhs_ptr + 4 * i);
    //   }
    //   for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < RhsCells; j++) {
    //       acc[i][4 * j + 0] = vmlaq_lane_f32(acc[i][4 * j + 0], lhs[i],
    //                                          vget_low_f32(rhs[j]), 0);
    //       acc[i][4 * j + 1] = vmlaq_lane_f32(acc[i][4 * j + 1], lhs[i],
    //                                          vget_low_f32(rhs[j]), 1);
    //       acc[i][4 * j + 2] = vmlaq_lane_f32(acc[i][4 * j + 2], lhs[i],
    //                                          vget_high_f32(rhs[j]), 0);
    //       acc[i][4 * j + 3] = vmlaq_lane_f32(acc[i][4 * j + 3], lhs[i],
    //                                          vget_high_f32(rhs[j]), 1);
    //     }
    //   }
    //   lhs_ptr += 12;
    //   rhs_ptr += 4 * RhsCells;
    // }

    // for (int i = 0; i < 3; i++) {
    //   for (int j = 0; j < 4 * RhsCells; j++) {
    //     vst1q_f32(accum_ptr + 4 * (i + 3 * j), acc[i][j]);
    //   }
    // }

    // for (int i = 0; i < 3; i++) {
    //   for (int j = 0; j < 4 * RhsCells; j++) {
    //     vst1q_f32(accum_ptr + 4 * (i + 3 * j), acc[i][j]);
    //     std::cout << "ACC("<< i << ", " << j << ") =" <<  accum_ptr + 4 * (i + 3 * j) << std::endl;
    //   }
    // }
  }
};

#endif

#if RUY_PLATFORM(NEON_32)

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BgemmKernel<ruy::Path::kNeon, LhsScalar, RhsScalar, DstScalar, Spec> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to binary kernel should have the same type.");
    static_assert(
        // std::is_unsigned<LhsScalar>::value &&
        std::is_integral<LhsScalar>::value,
        "Input to binary kernel should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");

    // TODO: not implemented -> fallback to standard cpp
  }
};

#endif

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BgemmKernel<ruy::Path::kNeonDotprod, LhsScalar, RhsScalar, DstScalar,
                   Spec> {
  ruy::Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using Base = BgemmKernel<Path::kNeon, LhsLayout, RhsLayout, DstScalar, Spec>;
  explicit BgemmKernel(ruy::Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to binary kernel should have the same type.");
    static_assert(
        /* std::is_unsigned<LhsScalar>::value && */
        std::is_integral<LhsScalar>::value,
        "Input to binary kernel should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    // TODO: not implemented
  }
};

#endif  // RUY_PLATFORM(NEON) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_
