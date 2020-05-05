#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

#include "bgemm_kernels_common.h"
#include "bgemm_trmul_params.h"
#include "ruy/context_internal.h"
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
    auto ruy_context = context->ruy_context();

    // Set up the matrix layouts and spec.
    ruy::Matrix<LhsScalar> lhs;
    ruy::MakeSimpleLayout(lhs_params.rows, lhs_params.cols,
                          ruy::Order::kRowMajor, &lhs.layout);
    ruy::Matrix<RhsScalar> rhs;
    ruy::MakeSimpleLayout(rhs_params.rows, rhs_params.cols,
                          ruy::Order::kColMajor, &rhs.layout);
    ruy::Matrix<DstScalar> dst;
    ruy::MakeSimpleLayout(dst_params.rows, dst_params.cols,
                          ruy::Order::kColMajor, &dst.layout);
    lhs.data = lhs_data;
    rhs.data = rhs_data;
    dst.data = dst_data;

    BinaryBasicSpec<AccumScalar, DstScalar> spec;
    spec.output_transform = output_transform;

    // The allocator is used to allocate memory for pre-packed matrices
    ruy::Allocator allocator;
    auto alloc_fn = [&allocator](std::size_t num_bytes) -> void* {
      return allocator.AllocateBytes(num_bytes);
    };

    constexpr auto BGemmCompiledPaths = ruy::kAllPaths & ~ruy::Path::kReference;

    // avoid the reference path for production code
    ruy::Path bgemm_runtime_path =
        ContextInternal::GetPathToTake<ruy::kAllPaths>(ruy_context);
    RUY_CHECK_NE(bgemm_runtime_path, ruy::Path::kReference);

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

    ruy::Matrix<LhsScalar> transposed_lhs(lhs);
    Transpose(&transposed_lhs);

    ruy::TrMulParams binary_trmul_params;
    CreateBinaryTrMulParams<BGemmCompiledPaths>(
        transposed_lhs, rhs, spec, ruy_context, &dst, bgemm_runtime_path,
        &binary_trmul_params);

    // pre-pack the lhs and rhs matrices
    ruy::PrepackedMatrix prepacked_lhs;
    ruy::PrepackedMatrix prepacked_rhs;
    ruy::SidePair<ruy::PrepackedMatrix*> prepacked(&prepacked_lhs,
                                                   &prepacked_rhs);

    const ruy::SidePair<int> origin{0, 0};
    const ruy::SidePair<int> rounded_dims{
        binary_trmul_params.packed[ruy::Side::kLhs].layout.cols,
        binary_trmul_params.packed[ruy::Side::kRhs].layout.cols};

    ruy::Tuning tuning = ContextInternal::GetMainThreadTuning(ruy_context);
    for (ruy::Side side : {ruy::Side::kLhs, ruy::Side::kRhs}) {
      if (prepacked[side]) {
        prepacked[side]->data_size = DataSize(binary_trmul_params.packed[side]);
        prepacked[side]->sums_size = SumsSize(binary_trmul_params.packed[side]);
        prepacked[side]->data = alloc_fn(prepacked[side]->data_size);
        prepacked[side]->sums = alloc_fn(prepacked[side]->sums_size);
        binary_trmul_params.packed[side].data = prepacked[side]->data;
        binary_trmul_params.packed[side].sums = prepacked[side]->sums;
        binary_trmul_params.RunPack(side, tuning, origin[side],
                                    rounded_dims[side]);
        binary_trmul_params.is_prepacked[side] = true;
      }
    }

    ruy::TrMul(&binary_trmul_params, ruy_context);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
