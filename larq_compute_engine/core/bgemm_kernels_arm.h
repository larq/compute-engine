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

#if RUY_PLATFORM(NEON)

// Generic kNeon template when no types are specified
template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BgemmKernel<ruy::Path::kNeon, LhsScalar, RhsScalar, DstScalar, Spec> {
  ruy::Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(ruy::Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    TFLITE_DCHECK(false);
  }
};

// Generic kNeonDotprod template
template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BgemmKernel<ruy::Path::kNeonDotprod, LhsScalar, RhsScalar, DstScalar,
                   Spec> {
  ruy::Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(ruy::Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    TFLITE_DCHECK(false);
  }
};

#if RUY_PLATFORM(NEON) && RUY_OPT_ENABLED(RUY_OPT_ASM) && RUY_PLATFORM(NEON_64)
// A BGEMM kernel for ARM64 Neon.
#include "bgemm_kernels_arm64.h"

/******************************
 * Kernels for uint64 inputs. *
 ******************************/

// Optimised kernel for float output with 16-bit accumulators.
template <>
struct BgemmKernel<ruy::Path::kNeon, std::uint64_t, std::uint64_t, float,
                   BinaryBasicSpec<std::int16_t, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 2, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 2, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<std::uint64_t>& lhs,
           const ruy::PackedMatrix<std::uint64_t>& rhs,
           const BinaryBasicSpec<std::int16_t /* accum. scalar */, float>& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<float>* dst) const {
    BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint64_t>
        params;
    MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
                           end_col, dst, &params);
    BinaryKernelNeonOutOfOrder8x4(params);
  }
};

// Fallback kernel for float output with 32-bit accumulators (when there's a
// risk of overflowing 16-bit accumulators).
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
    BinaryKernelNeonOutOfOrder4x4(params);
  }
};

/******************************
 * Kernels for uint32 inputs. *
 ******************************/

// With uint32 inputs, we can still use the optimised kernels designed for
// uint64 inputs by making adjustments to the declared kernel depth and LHS/RHS
// strides (the latter is done in a specialised template of
// `MakeBinaryKernelParams` that accepts uint32 LHS/RHS and uint64
// `BinaryKernelParams`).

// Optimised kernel for float output with 16-bit accumulators.
template <>
struct BgemmKernel<ruy::Path::kNeon, std::uint32_t, std::uint32_t, float,
                   BinaryBasicSpec<std::int16_t, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<std::uint32_t>& lhs,
           const ruy::PackedMatrix<std::uint32_t>& rhs,
           const BinaryBasicSpec<std::int16_t /* accum. scalar */, float>& spec,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Matrix<float>* dst) const {
    BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint64_t>
        params;
    MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
                           end_col, dst, &params);
    BinaryKernelNeonOutOfOrder8x4(params);
  }
};

// Fallback kernel for float output with 32-bit accumulators (when there's a
// risk of overflowing 16-bit accumulators).
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
    BinaryKernelParams<LhsLayout::kCols, RhsLayout::kCols, std::uint64_t>
        params;
    MakeBinaryKernelParams(lhs, rhs, spec, start_row, start_col, end_row,
                           end_col, dst, &params);
    BinaryKernelNeonOutOfOrder4x4(params);
  }
};

#endif  // RUY_OPT_ENABLED(RUY_OPT_ASM) && RUY_PLATFORM(NEON_64)

#endif  // RUY_PLATFORM(NEON)

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_
