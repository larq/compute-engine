#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

#include "bgemm_kernels_ruy.h"
#include "tensorflow/lite/experimental/ruy/dispatch.h"
#include "tensorflow/lite/experimental/ruy/ruy_advanced.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

// Simple allocator for allocating pre-packed matrices.
class SimpleAllocator {
 public:
  void* AllocateBytes(std::size_t num_bytes) {
    char* p = new char[num_bytes];
    buffers_.emplace_back(p);
    return static_cast<void*>(p);
  }

 private:
  std::vector<std::unique_ptr<char[]>> buffers_;
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct BGemmImplUsingRuy {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    gemmlowp::ScopedProfilingLabel label("BGemmRuy");

    static_assert(std::is_same<LhsScalar, RhsScalar>::value,
                  "Inputs to BGEMM should have the same type.");
    static_assert(std::is_unsigned<LhsScalar>::value &&
                      std::is_integral<LhsScalar>::value,
                  "Input to BGEMM should be of type unsigned integral.");
    static_assert(std::is_signed<DstScalar>::value,
                  "Output of BGEMM should be of a signed type.");

    using T = LhsScalar;
    // default accumulator bitwidth is set to 32-bit which is enough for
    // bitwidths we are currently using
    using TAccum = std::int32_t;
    using TSpec = ruy::BasicSpec<TAccum, DstScalar>;

    // getting ruy context
    auto ruy_context = context->ruy_context();

    // TODO: binary path should be determined on compile time
    constexpr auto binary_path = ruy::Path::kStandardCpp;

    // avoid the reference path for production code
    ruy::Path the_path = ruy_context->GetPathToTake<ruy::kAllPaths>();
    RUY_CHECK_NE(the_path, ruy::Path::kReference);

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

    // Here, we abuse the 'multiplier_exponent' which is used only for
    // non-floating-point cases to pass the bitpadding correction value (int) to
    // bgemm kernel
    TSpec spec;
    spec.multiplier_exponent = params.multiplier_exponent;

    // The allocator is used to allocate memory for pre-packed matrices
    // TODO: needs aligned allocator for NEON SIMD?
    SimpleAllocator allocator;
    auto alloc_fn = [&allocator](std::size_t num_bytes) -> void* {
      return allocator.AllocateBytes(num_bytes);
    };

    // pre-pack the lhs and rhs matrices
    ruy::PrepackedMatrix prepacked_lhs;
    ruy::PrepackedMatrix prepacked_rhs;
    ruy::PrePackForMul<binary_path>(lhs, rhs, spec, ruy_context, &dst,
                                    &prepacked_lhs, &prepacked_rhs, alloc_fn);

    // In Ruy, TrMul is computed instead of Mul, therefore the lhs needs to be
    // transposed. Transpose function is cheap since it does not shuffle data
    // around and only changes the matrix layout.
    ruy::Matrix<LhsScalar> transposed_lhs(lhs);
    ruy::Transpose(&transposed_lhs);

    // Based on the Path, kernel function pointers are set in TrMulParams
    // constexpr ruy::Path TrMulCompiledPaths =
    //     ruy::kAllPaths & ~ruy::Path::kReference;
    ruy::TrMulParams trmul_params;
    ruy::CreateTrMulParams<binary_path>(transposed_lhs, rhs, spec, ruy_context,
                                        &dst, binary_path, &trmul_params);

    // set the pre-packed params
    trmul_params.packed[ruy::Side::kLhs].data = prepacked_lhs.data;
    trmul_params.packed[ruy::Side::kLhs].sums = prepacked_lhs.sums;
    trmul_params.is_prepacked[ruy::Side::kLhs] = true;

    trmul_params.packed[ruy::Side::kRhs].data = prepacked_rhs.data;
    trmul_params.packed[ruy::Side::kRhs].sums = prepacked_rhs.sums;
    trmul_params.is_prepacked[ruy::Side::kRhs] = true;

    // redirect the kernel function pointer to the binary kernel
    using PackedLhsScalar = ruy::PackedType<binary_path, LhsScalar>;
    using PackedRhsScalar = ruy::PackedType<binary_path, RhsScalar>;
    trmul_params.run_kernel =
        &RunBgemmKernel<binary_path, PackedLhsScalar, PackedRhsScalar,
                        DstScalar, TSpec>;

    ruy::TrMul(&trmul_params, ruy_context);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
