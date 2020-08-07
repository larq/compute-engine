#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_

#include "larq_compute_engine/core/types.h"
#include "ruy/common.h"
#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/tune.h"

using namespace ruy;

using compute_engine::core::TBitpacked;

#if RUY_PLATFORM_X86

template <typename DstScalar, typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvx2, DstScalar, MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

template <typename DstScalar, typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvx512, DstScalar, MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

template <typename DstScalar, typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvxVnni, DstScalar, MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

template <typename DstScalar, typename MulParamsType>
struct BgemmKernel<Path::kSse42, DstScalar, MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PMat<TBitpacked>& lhs, const ruy::PMat<TBitpacked>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Mat<DstScalar>* dst) const {
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

#endif  // RUY_PLATFORM_X86

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_
