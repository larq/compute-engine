#ifndef COMPUTE_ENGINE_CORE_BGEMM_RUY_TRMUL_PARAMS_H_
#define COMPUTE_ENGINE_CORE_BGEMM_RUY_TRMUL_PARAMS_H_

#include "larq_compute_engine/core/bgemm/kernels.h"
#include "larq_compute_engine/core/bgemm/ruy_pack.h"
#include "ruy/create_trmul_params.h"
#include "ruy/mul_params.h"
#include "ruy/path.h"
#include "ruy/trmul_params.h"

namespace compute_engine {
namespace core {
namespace bgemm {

inline bool IsColMajorTrMul(const ruy::TrMulParams& params) {
  return IsColMajor(params.src[Side::kLhs].layout) &&
         IsColMajor(params.src[Side::kRhs].layout) &&
         IsColMajor(params.dst.layout);
}

template <ruy::Path ThePath, typename DstScalar, typename MulParamsType>
void PopulateBGemmTrMulParams(const Mat<TBitpacked>& lhs,
                              const Mat<TBitpacked>& rhs, Mat<DstScalar>& dst,
                              const MulParamsType& mul_params,
                              ruy::TrMulParams* params) {
  params->src[Side::kLhs] = EraseType(lhs);
  params->src[Side::kRhs] = EraseType(rhs);
  params->dst = EraseType(dst);

  static_assert(alignof(MulParamsType) <= kMaxMulParamsAlignment, "");
  static_assert(sizeof(MulParamsType) <= kMaxMulParamsSize, "");
  static_assert(std::is_trivially_copyable<MulParamsType>::value, "");
  auto* dst_mul_params =
      reinterpret_cast<MulParamsType*>(params->mul_params_bytes);
  std::memcpy(dst_mul_params, &mul_params, sizeof(MulParamsType));

  // Optimised code paths only support all matrices being column-major
  if (!IsColMajorTrMul(*params) && ThePath != ruy::Path::kStandardCpp) {
    PopulateBGemmTrMulParams<ruy::Path::kStandardCpp>(lhs, rhs, dst, mul_params,
                                                      params);
    return;
  };

  using Kernel = BGemmKernel<ThePath, DstScalar, MulParamsType>;
  using LhsKernelLayout = typename Kernel::LhsLayout;
  using RhsKernelLayout = typename Kernel::RhsLayout;

  params->path = ThePath;

  ruy::detail::CreatePackedMatrix<TBitpacked, TBitpacked>(
      Side::kLhs, ToKernelLayout<LhsKernelLayout>(), params);
  ruy::detail::CreatePackedMatrix<TBitpacked, TBitpacked>(
      Side::kRhs, ToKernelLayout<RhsKernelLayout>(), params);
  params->run_pack[Side::kLhs] = &RunRuyPack<ThePath, LhsKernelLayout>;
  params->run_pack[Side::kRhs] = &RunRuyPack<ThePath, RhsKernelLayout>;
  params->run_kernel = &RunBGemmKernel<ThePath, DstScalar, MulParamsType>;
}

}  // namespace bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BGEMM_RUY_TRMUL_PARAMS_H_
