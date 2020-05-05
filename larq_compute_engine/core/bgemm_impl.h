#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_

#include "bgemm_kernels_common.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

// TODO: currently only ref. impl. is supported
#ifndef TFLITE_WITH_RUY
#include "bgemm_impl_ref.h"
#else
#include "bgemm_impl_ruy.h"
#endif

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

#ifndef TFLITE_WITH_RUY
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct BGemmImpl : BGemmImplRef<LhsScalar, RhsScalar, AccumScalar, DstScalar> {
};
#else
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct BGemmImpl
    : BGemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar> {};
#endif

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
void BGemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
           const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
           const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
           const OutputTransform<AccumScalar, DstScalar>& params,
           CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("BGemm");
  // TODO: special fast bgemm impl. for matrix-vector multiplication
  // if (dst_params.cols == 1) {
  //   // GEMV case: try a custom fast GEMV path.
  //   if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
  //                          dst_params, dst_data, params, context)) {
  //     return;
  //   }
  // }
  ruy::profiler::ScopeLabel label2("BGemm/GeneralBGEMM");
  BGemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar>::Run(
      lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data, params,
      context);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_
