#ifndef COMPUTE_ENGINE_CORE_BGEMM_IMPL_H_
#define COMPUTE_ENGINE_CORE_BGEMM_IMPL_H_

#include "bgemm_kernels_common.h"
#include "bgemm_trmul_params.h"
#include "ruy/context.h"
#include "ruy/context_get_ctx.h"
#include "ruy/matrix.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

using compute_engine::core::TBitpacked;

template <typename AccumScalar, typename DstScalar>
void BGemm(const MatrixParams<TBitpacked>& lhs_params,
           const TBitpacked* lhs_data,
           const MatrixParams<TBitpacked>& rhs_params,
           const TBitpacked* rhs_data,
           const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
           const OutputTransform<DstScalar>& output_transform,
           CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("BGemm (Ruy)");

  static_assert(std::is_signed<DstScalar>::value,
                "Output of BGEMM should be of a signed type.");

  // Get ruy context
  auto ruy_ctx = get_ctx(context->ruy_context());

  // Set up the matrix layouts and mul_params.
  ruy::Matrix<TBitpacked> lhs;
  ruy::Matrix<TBitpacked> rhs;
  ruy::Matrix<DstScalar> dst;
  // We allow these matrices to be cached. Note that this doesn't force them
  // to be cached; it means that the `cache_policy` of the MatrixParams will
  // be respected.
  cpu_backend_gemm::detail::MakeRuyMatrix(lhs_params, lhs_data, &lhs,
                                          /*use_caching=*/true);
  cpu_backend_gemm::detail::MakeRuyMatrix(rhs_params, rhs_data, &rhs,
                                          /*use_caching=*/true);
  cpu_backend_gemm::detail::MakeRuyMatrix(dst_params, dst_data, &dst);

  // We have to make this a `const` matrix because otherwise gcc will try to
  // use the non-const versions of `matrix.data()`
  ruy::Mat<TBitpacked> internal_lhs =
      ruy::ToInternal((const ruy::Matrix<TBitpacked>)lhs);
  ruy::Mat<TBitpacked> internal_rhs =
      ruy::ToInternal((const ruy::Matrix<TBitpacked>)rhs);
  ruy::Mat<DstScalar> internal_dst = ruy::ToInternal(dst);

  BinaryMulParams<AccumScalar, DstScalar> mul_params;
  mul_params.output_transform = output_transform;

#if RUY_PLATFORM_NEON
  constexpr bool HasOptimizedNeonKernel =
      std::is_same<AccumScalar, std::int16_t>::value ||
      std::is_same<DstScalar, float>::value ||
      std::is_same<DstScalar, std::int8_t>::value;
  constexpr auto SelectedPath =
      HasOptimizedNeonKernel ? ruy::Path::kNeon : ruy::Path::kStandardCpp;
#else
  constexpr auto SelectedPath = ruy::Path::kStandardCpp;
#endif

  ruy::Mat<TBitpacked> transposed_lhs(internal_lhs);
  Transpose(&transposed_lhs);

  ruy::TrMulParams bgemm_trmul_params;
  PopulateBGemmTrMulParams<SelectedPath>(transposed_lhs, internal_rhs,
                                         internal_dst, mul_params,
                                         &bgemm_trmul_params);

  HandlePrepackedCaching(&bgemm_trmul_params, ruy_ctx);
  ruy::TrMul(&bgemm_trmul_params, ruy_ctx);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_IMPL_H_
