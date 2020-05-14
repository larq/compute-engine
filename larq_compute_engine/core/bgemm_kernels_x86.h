#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_

#include "ruy/common.h"
#include "ruy/internal_matrix.h"
#include "ruy/kernel_common.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/tune.h"

using namespace ruy;

#if RUY_PLATFORM(X86)

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvx2, LhsScalar, RhsScalar, DstScalar,
                   MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Matrix<DstScalar>* dst) const {
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to binary kernel should have the same type.");
    static_assert(
        // std::is_unsigned<LhsScalar>::value &&
        std::is_integral<LhsScalar>::value,
        "Input to binary kernel should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvx512, LhsScalar, RhsScalar, DstScalar,
                   MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Matrix<DstScalar>* dst) const {
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to binary kernel should have the same type.");
    static_assert(
        // std::is_unsigned<LhsScalar>::value &&
        std::is_integral<LhsScalar>::value,
        "Input to binary kernel should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of binary kernel should be of a signed type.");
    TFLITE_DCHECK(false);
  }
};

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct BgemmKernel<ruy::Path::kAvxVnni, LhsScalar, RhsScalar, DstScalar,
                   MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Matrix<DstScalar>* dst) const {
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

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct BgemmKernel<Path::kSse42, LhsScalar, RhsScalar, DstScalar,
                   MulParamsType> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit BgemmKernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, ruy::Matrix<DstScalar>* dst) const {
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

#endif  // RUY_PLATFORM(X86)

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_X86_H_
