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

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct BGemmImplUsingRuy {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const OutputTransform<AccumScalar, DstScalar>& output_transform,
      CpuBackendContext* context) {
    ruy::profiler::ScopeLabel label("BGemmRuy");

    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to BGEMM should have the same type.");
    static_assert(std::is_unsigned<LhsScalar>::value &&
                      std::is_integral<LhsScalar>::value,
                  "Input to BGEMM should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of BGEMM should be of a signed type.");

    // getting ruy context
    auto ruy_ctx = get_ctx(context->ruy_context());

    // Set up the matrix layouts and mul_params.
    ruy::Matrix<LhsScalar> lhs;
    ruy::MakeSimpleLayout(lhs_params.rows, lhs_params.cols,
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    ruy::Matrix<RhsScalar> rhs;
    ruy::MakeSimpleLayout(rhs_params.rows, rhs_params.cols,
                          ruy::Order::kColMajor, rhs.mutable_layout());
    ruy::Matrix<DstScalar> dst;
    ruy::MakeSimpleLayout(dst_params.rows, dst_params.cols,
                          ruy::Order::kColMajor, dst.mutable_layout());
    lhs.set_data(lhs_data);
    rhs.set_data(rhs_data);
    dst.set_data(dst_data);

    // We have to make this a `const` matrix because otherwise gcc will try to
    // use the non-const versions of `matrix.data()`
    ruy::Mat<LhsScalar> internal_lhs =
        ruy::ToInternal((const ruy::Matrix<LhsScalar>)lhs);
    ruy::Mat<RhsScalar> internal_rhs =
        ruy::ToInternal((const ruy::Matrix<RhsScalar>)rhs);
    ruy::Mat<DstScalar> internal_dst = ruy::ToInternal(dst);

    BinaryMulParams<AccumScalar, DstScalar> mul_params;
    mul_params.output_transform = output_transform;

    constexpr auto BGemmCompiledPaths = ruy::kAllPaths;

    // avoid the reference path for production code
    ruy::Path bgemm_runtime_path = ruy_ctx->SelectPath(ruy::kAllPaths);

    // fallback to standard cpp kernel for all architectures that are not
    // supported yet.
    // TODO: this needs to be modified as soon as architecture-specific
    // optimized kernels are added.
#if RUY_PLATFORM(ARM)
    if (bgemm_runtime_path == ruy::Path::kNeonDotprod)
      bgemm_runtime_path = ruy::Path::kNeon;
#if RUY_PLATFORM(NEON_32)
    // 32-bit NEON optimized code is not available yet
    bgemm_runtime_path = ruy::Path::kStandardCpp;
#endif
    // Currently we only have 32-bit and 64-bit optimized kernels.
    // For 8-bit, fall back to the standard cpp kernel.
    if (std::is_same<LhsScalar, std::uint8_t>::value)
      bgemm_runtime_path = ruy::Path::kStandardCpp;
#elif RUY_PLATFORM(X86)
    if (bgemm_runtime_path == ruy::Path::kAvx2 ||
        bgemm_runtime_path == ruy::Path::kAvx512)
      bgemm_runtime_path = ruy::Path::kStandardCpp;
#endif

    // For writing bitpacked output, fallback to the standard C++ kernel.
    if (std::is_same<DstScalar, std::int32_t>::value) {
      bgemm_runtime_path = ruy::Path::kStandardCpp;
    }

    // For now, int8 only has a C++ implementation
    if (std::is_same<DstScalar, std::int8_t>::value) {
      bgemm_runtime_path = ruy::Path::kStandardCpp;
    }

    ruy::Mat<LhsScalar> transposed_lhs(internal_lhs);
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
