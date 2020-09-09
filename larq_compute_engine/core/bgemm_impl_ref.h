#ifndef COMPUTE_ENGINE_CORE_BGEMM_IMPL_REF_H_
#define COMPUTE_ENGINE_CORE_BGEMM_IMPL_REF_H_

#include "larq_compute_engine/core/bgemm_functor.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

using ce::core::TBitpacked;

template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct BGemmImplRef {
  static void Run(
      const MatrixParams<TBitpacked>& lhs_params, const TBitpacked* lhs_data,
      const MatrixParams<TBitpacked>& rhs_params, const TBitpacked* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    ruy::profiler::ScopeLabel label("BGemmRef");

    static_assert(std::is_signed<DstScalar>::value,
                  "Output of BGEMM should be of a signed type.");

    // This code assumes specific memory layout
    // assert(rhs_params.order == cpu_backend_gemm::Order::kColMajor);
    using TBGemmFunctor =
        ce::core::ReferenceBGemmFunctor<ce::core::Layout::RowMajor,
                                        ce::core::Layout::ColMajor, DstScalar,
                                        ce::core::Layout::ColMajor>;

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

#endif  // COMPUTE_ENGINE_CORE_BGEMM_IMPL_REF_H_
