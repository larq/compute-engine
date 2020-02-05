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

#if RUY_PLATFORM(NEON_64)
// A BGEMM kernel for ARM64 Neon.
#include "bgemm_kernels_arm64.h"

// specialized kernel for 32-bit bitpacking, float output and 32-bit accumulator
template <>
struct BgemmKernel<ruy::Path::kNeon, std::uint32_t, std::uint32_t, float,
                   BinaryBasicSpec<std::int32_t, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<std::uint32_t>& lhs,
           const ruy::PackedMatrix<std::uint32_t>& rhs,
           const BinaryBasicSpec<std::int32_t /* accum. scalar */, float>& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<float>* dst) const {
    BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint32_t>
        params;
    MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
                           end_col, dst, &params);
    BinaryKernelNeonOutOfOrder32BP4x4(params);
  }
};

// // specialized kernel for 64-bit bitpacking, float output and 32-bit accumulator
// template <>
// struct BgemmKernel<ruy::Path::kNeon, std::uint64_t, std::uint64_t, float,
//                    BinaryBasicSpec<std::int32_t, float>> {
//   Tuning tuning = Tuning::kAuto;
//   using LhsLayout = FixedKernelLayout<Order::kColMajor, 2, 4>;
//   using RhsLayout = FixedKernelLayout<Order::kColMajor, 2, 4>;
//   explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
//   void Run(const ruy::PackedMatrix<std::uint64_t>& lhs,
//            const ruy::PackedMatrix<std::uint64_t>& rhs,
//            const BinaryBasicSpec<std::int32_t /* accum. scalar */, float>& spec,
//            int start_row, int start_col, int end_row, int end_col,
//            ruy::Matrix<float>* dst) const {
//     BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint64_t>
//         params;
//     MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
//                            end_col, dst, &params);
//     BinaryKernelNeonOutOfOrder64BP4x4(params);
//   }
// };

// specialized kernel for 64-bit bitpacking, float output and 32-bit accumulator
template <>
struct BgemmKernel<ruy::Path::kNeon, std::uint64_t, std::uint64_t, float,
                   BinaryBasicSpec<std::int32_t, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 2, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 2, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<std::uint64_t>& lhs,
           const ruy::PackedMatrix<std::uint64_t>& rhs,
           const BinaryBasicSpec<std::int32_t /* accum. scalar */, float>& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<float>* dst) const {
    BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint64_t>
        params;
    MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
                           end_col, dst, &params);
    BinaryKernelNeonOutOfOrder64BP4x4D6(params);
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
