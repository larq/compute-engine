#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

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
struct BGemmImplUsingRuy {
  static void Run(
      const MatrixParams<TBitpacked>& lhs_params, const TBitpacked* lhs_data,
      const MatrixParams<TBitpacked>& rhs_params, const TBitpacked* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const OutputTransform<AccumScalar, DstScalar>& output_transform,
      CpuBackendContext* context) {
    ruy::profiler::ScopeLabel label("BGemmRuy");

    static_assert(std::is_signed<DstScalar>::value,
                  "Output of BGEMM should be of a signed type.");

    // getting ruy context
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

    constexpr auto BGemmCompiledPaths = ruy::kAllPaths;

    ruy::Path bgemm_runtime_path = ruy_ctx->SelectPath(ruy::kAllPaths);

#if RUY_PLATFORM_ARM
    if (bgemm_runtime_path == ruy::Path::kNeonDotprod)
      bgemm_runtime_path = ruy::Path::kNeon;
#elif RUY_PLATFORM_X86
    if (bgemm_runtime_path == ruy::Path::kAvx2 ||
        bgemm_runtime_path == ruy::Path::kAvx512)
      bgemm_runtime_path = ruy::Path::kStandardCpp;
#endif

    // For now, writing bitpacked output only has a C++ implementation.
    if (std::is_same<DstScalar, TBitpacked>::value) {
      bgemm_runtime_path = ruy::Path::kStandardCpp;
    }

    // For now, int8 output only has a C++ implementation.
    if (std::is_same<DstScalar, std::int8_t>::value) {
      bgemm_runtime_path = ruy::Path::kStandardCpp;
    }

    ruy::Mat<TBitpacked> transposed_lhs(internal_lhs);
    Transpose(&transposed_lhs);

    ruy::TrMulParams binary_trmul_params;
    CreateBinaryTrMulParams<BGemmCompiledPaths>(
        transposed_lhs, internal_rhs, mul_params, ruy_ctx, &internal_dst,
        bgemm_runtime_path, &binary_trmul_params);

    HandlePrepackedCaching(&binary_trmul_params, ruy_ctx);
    ruy::TrMul(&binary_trmul_params, ruy_ctx);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
