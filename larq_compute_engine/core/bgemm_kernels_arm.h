#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_

#include "bgemm_kernels_common.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/common.h"
#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/side_pair.h"
#include "ruy/size_util.h"
#include "ruy/tune.h"

using namespace ruy;

using compute_engine::core::TBitpacked;

#if RUY_PLATFORM_NEON && RUY_OPT(ASM) && RUY_PLATFORM_NEON_32
// A BGEMM kernel for ARM32 Neon.
#include "bgemm_kernels_arm32.h"

// Optimised Arm32 kernel. Supports float or int8 output.
template <typename DstScalar>
struct BgemmKernel<ruy::Path::kNeon, DstScalar,
                   BinaryMulParams<std::int32_t, DstScalar>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const BinaryMulParams<std::int32_t, DstScalar>& mul_params,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_same<DstScalar, float>::value ||
                      std::is_same<DstScalar, std::int8_t>::value,
                  "");
    BinaryKernelParams<DstScalar, LhsLayout::kCols, RhsLayout::kCols> params;
    MakeBinaryKernelParams(lhs, rhs, start_row, start_col, end_row, end_col,
                           dst, mul_params, &params);
    BinaryKernelNeonOutOfOrder4x4(params);
  }
};

#endif

#if RUY_PLATFORM_NEON && RUY_OPT(ASM) && RUY_PLATFORM_NEON_64
// A BGEMM kernel for ARM64 Neon.
#include "bgemm_kernels_arm64.h"

// Optimised Aarch64 kernel with 16-bit accumulators. Supports float or int8 or
// bitpacked output.
template <typename DstScalar>
struct BgemmKernel<ruy::Path::kNeon, DstScalar,
                   BinaryMulParams<std::int16_t, DstScalar>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const BinaryMulParams<std::int16_t, DstScalar>& mul_params,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_same<DstScalar, float>::value ||
                      std::is_same<DstScalar, std::int8_t>::value ||
                      std::is_same<DstScalar, TBitpacked>::value,
                  "");
    BinaryKernelParams<DstScalar, LhsLayout::kCols, RhsLayout::kCols> params;
    MakeBinaryKernelParams(lhs, rhs, start_row, start_col, end_row, end_col,
                           dst, mul_params, &params);
    BinaryKernelNeonOutOfOrder8x4(params);
  }
};

// Fallback Aarch64 kernel with 32-bit accumulators (when there's a risk of
// overflowing 16-bit accumulators). Supports float or int8 output.
template <typename DstScalar>
struct BgemmKernel<ruy::Path::kNeon, DstScalar,
                   BinaryMulParams<std::int32_t, DstScalar>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const BinaryMulParams<std::int32_t, DstScalar>& mul_params,
           int start_row, int start_col, int end_row, int end_col,
           ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_same<DstScalar, float>::value ||
                      std::is_same<DstScalar, std::int8_t>::value,
                  "");
    BinaryKernelParams<DstScalar, LhsLayout::kCols, RhsLayout::kCols> params;
    MakeBinaryKernelParams(lhs, rhs, start_row, start_col, end_row, end_col,
                           dst, mul_params, &params);
    BinaryKernelNeonOutOfOrder4x4(params);
  }
};

#endif  // RUY_OPT(ASM) && RUY_PLATFORM_NEON_64

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_ARM_H_
