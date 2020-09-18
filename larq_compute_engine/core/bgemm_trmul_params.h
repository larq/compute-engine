#ifndef COMPUTE_ENGINE_CORE_BGEMM_TRMUL_PARAMS_H_
#define COMPUTE_ENGINE_CORE_BGEMM_TRMUL_PARAMS_H_

#include "larq_compute_engine/core/bgemm_kernels_ruy.h"
#include "larq_compute_engine/core/ruy_pack.h"
#include "ruy/dispatch.h"
#include "ruy/mul_params.h"
#include "ruy/path.h"
#include "ruy/trmul_params.h"

namespace compute_engine {
namespace tflite {

template <ruy::Path ThePath, typename DstScalar, typename MulParamsType>
void PopulateBGemmTrMulParams(const Mat<TBitpacked>& lhs,
                              const Mat<TBitpacked>& rhs, Mat<DstScalar>& dst,
                              const MulParamsType& mul_params,
                              ruy::TrMulParams* params) {
  params->src[Side::kLhs] = EraseType(lhs);
  params->src[Side::kRhs] = EraseType(rhs);
  params->dst = EraseType(dst);
  params->mul_params = ToVoidPtr(&mul_params);

  // Optimised code paths only support all matrices being column-major
  if (!ruy::IsColMajorTrMul(*params) && ThePath != ruy::Path::kStandardCpp) {
    PopulateBGemmTrMulParams<ruy::Path::kStandardCpp>(lhs, rhs, dst, mul_params,
                                                      params);
    return;
  };

  using Kernel = BGemmKernel<ThePath, DstScalar, MulParamsType>;
  using LhsKernelLayout = typename Kernel::LhsLayout;
  using RhsKernelLayout = typename Kernel::RhsLayout;

  params->path = ThePath;

  CreatePackedMatrix<TBitpacked, TBitpacked>(
      Side::kLhs, ToKernelLayout<LhsKernelLayout>(), params);
  CreatePackedMatrix<TBitpacked, TBitpacked>(
      Side::kRhs, ToKernelLayout<RhsKernelLayout>(), params);
  params->run_pack[Side::kLhs] =
      &compute_engine::tflite::RunPack<ThePath, LhsKernelLayout>;
  params->run_pack[Side::kRhs] =
      &compute_engine::tflite::RunPack<ThePath, RhsKernelLayout>;
  params->run_kernel = &RunBGemmKernel<ThePath, DstScalar, MulParamsType>;
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_TRMUL_PARAMS_H_
