#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct BGemmImplRef {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    gemmlowp::ScopedProfilingLabel label("BGemmRef");

    // these checkes will be also done in BGEMM functor but we do it here as
    // well to be on the safe side.
    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to BGEMM should have the same type.");
    static_assert(std::is_unsigned<LhsScalar>::value &&
                      std::is_integral<LhsScalar>::value,
                  "Input to BGEMM should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of BGEMM should be of a signed type.");

    using TBitpacked = LhsScalar;
    // This code assumes specific memory layout
    // assert(rhs_params.order == cpu_backend_gemm::Order::kColMajor);
    using TBGemmFunctor =
        ce::core::ReferenceBGemmFunctor<LhsScalar, ce::core::Layout::RowMajor,
                                        RhsScalar, ce::core::Layout::ColMajor,
                                        DstScalar, ce::core::Layout::ColMajor>;

    // LHS (n, k) -> RowMajor -> (n, k)
    // RHS (m, k) -> ColMajor -> (k, m)
    // DST (n, m) -> ColMajor -> (m, n)
    const auto n = lhs_params.rows;
    const auto k = lhs_params.cols;
    const auto m = rhs_params.cols;
    const auto lda = lhs_params.cols;
    // use number of rows for col-major layout
    const auto ldb = rhs_params.rows;
    const auto ldc = dst_params.rows;
    TBGemmFunctor bgemm_functor;
    // TODO: Currently GemmParmas is not used the same way as
    // as its used in the TF Lite codebase. Here, we abuse the
    // 'multiplier_exponent' which is used only for non-floating-point
    // cases to pass the bitpadding correction value (int) to BGemm
    bgemm_functor(n, m, k, lhs_data, lda, rhs_data, ldb, dst_data, ldc,
                  params.multiplier_exponent);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_
