#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

// TODO: currently only ref. impl. is supported
#ifndef TFLITE_WITH_RUY
#include "bgemm_impl_ref.h"
#else
#include "bgemm_impl_ref.h"
// #include "bgemm_impl_ruy.h"
#endif

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

#ifndef TFLITE_WITH_RUY
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct BGemmImpl : BGemmImplRef<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                quantization_flavor> {};
#else
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct BGemmImpl : BGemmImplRef<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                quantization_flavor> {};
// TODO: currently only ref. impl. is supported
// template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
//           typename DstScalar, QuantizationFlavor quantization_flavor>
// struct BGemmImpl : BGemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
//                                      DstScalar, quantization_flavor> {};
#endif

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void BGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
    CpuBackendContext* context) {
  gemmlowp::ScopedProfilingLabel label("bgemm_impl::BGemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  // TODO: special fast bgemm impl. for matrix-vector multiplication
  // if (dst_params.cols == 1) {
  //   // GEMV case: try a custom fast GEMV path.
  //   if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
  //                          dst_params, dst_data, params, context)) {
  //     return;
  //   }
  // }
  gemmlowp::ScopedProfilingLabel label2("bgemm_impl::BGemm: general BGEMM");
  BGemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
            quantization_flavor>::Run(lhs_params, lhs_data, rhs_params,
                                      rhs_data, dst_params, dst_data, params,
                                      context);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_IMPL_H_
