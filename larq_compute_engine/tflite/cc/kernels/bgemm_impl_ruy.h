#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

// #include "tensorflow/lite/experimental/ruy/dispatch.h"
// #include "tensorflow/lite/experimental/ruy/ruy_advanced.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
// #include "bgemm_kernels_ruy.h"
#include "bgemm_trmul_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

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

    // default accumulator bitwidth is set to 32-bit which is enough for
    // bitwidths we are currently using
    using TAccum = std::int32_t;
    using TSpec = ruy::BasicSpec<TAccum, DstScalar>;

    // getting ruy context
    auto ruy_context = context->ruy_context();

    // TODO: binary path should be determined on compile time
    // constexpr auto binary_path = ruy::Path::kAvx2;

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
    ruy::Allocator allocator;
    auto alloc_fn = [&allocator](std::size_t num_bytes) -> void* {
      return allocator.AllocateBytes(num_bytes);
    };

    constexpr auto TrMulCompiledPaths = ruy::kAllPaths & ~ruy::Path::kReference;

    // avoid the reference path for production code
    ruy::Path the_path = ruy_context->GetPathToTake<ruy::kAllPaths>();
    RUY_CHECK_NE(the_path, ruy::Path::kReference);

    ruy::Matrix<LhsScalar> transposed_lhs(lhs);
    Transpose(&transposed_lhs);

    ruy::TrMulParams binary_trmul_params;
    CreateBinaryTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, ruy_context, &dst,
                                                the_path, &binary_trmul_params);

    // pre-pack the lhs and rhs matrices
    ruy::PrepackedMatrix prepacked_lhs;
    ruy::PrepackedMatrix prepacked_rhs;
    ruy::SidePair<ruy::PrepackedMatrix*> prepacked(&prepacked_lhs, &prepacked_rhs);

    const ruy::SidePair<int> origin{0, 0};
    const ruy::SidePair<int> rounded_dims{binary_trmul_params.packed[ruy::Side::kLhs].layout.cols,
                                          binary_trmul_params.packed[ruy::Side::kRhs].layout.cols};

    ruy::Tuning tuning = ruy_context->GetMainThreadTuning();
    for (ruy::Side side : {ruy::Side::kLhs, ruy::Side::kRhs}) {
      if (prepacked[side]) {
        prepacked[side]->data_size = DataSize(binary_trmul_params.packed[side]);
        prepacked[side]->sums_size = SumsSize(binary_trmul_params.packed[side]);
        prepacked[side]->data = alloc_fn(prepacked[side]->data_size);
        prepacked[side]->sums = alloc_fn(prepacked[side]->sums_size);
        binary_trmul_params.packed[side].data = prepacked[side]->data;
        binary_trmul_params.packed[side].sums = prepacked[side]->sums;
        binary_trmul_params.RunPack(side, tuning, origin[side], rounded_dims[side]);
        binary_trmul_params.is_prepacked[side] = true;
      }
    }
    // }
    // // In Ruy, TrMul is computed instead of Mul, therefore the lhs needs to be
    // // transposed. Transpose function is cheap since it does not shuffle data
    // // around and only changes the matrix layout.
    // ruy::Matrix<LhsScalar> transposed_lhs(lhs);
    // ruy::Transpose(&transposed_lhs);

    // // Based on the Path, kernel function pointers are set in TrMulParams
    // // constexpr ruy::Path TrMulCompiledPaths =
    // //     ruy::kAllPaths & ~ruy::Path::kReference;
    // ruy::TrMulParams trmul_params;
    // ruy::CreateTrMulParams<binary_path>(transposed_lhs, rhs, spec, ruy_context,
    //                                     &dst, binary_path, &trmul_params);

    // set the pre-packed params
    // binary_trmul_params.packed[ruy::Side::kLhs].data = prepacked_lhs.data;
    // binary_trmul_params.packed[ruy::Side::kLhs].sums = prepacked_lhs.sums;
    // binary_trmul_params.is_prepacked[ruy::Side::kLhs] = true;

    // binary_trmul_params.packed[ruy::Side::kRhs].data = prepacked_rhs.data;
    // binary_trmul_params.packed[ruy::Side::kRhs].sums = prepacked_rhs.sums;
    // binary_trmul_params.is_prepacked[ruy::Side::kRhs] = true;

    // // redirect the kernel function pointer to the binary kernel
    // using PackedLhsScalar = ruy::PackedType<binary_path, LhsScalar>;
    // using PackedRhsScalar = ruy::PackedType<binary_path, RhsScalar>;
    // trmul_params.run_kernel =
    //     &RunBgemmKernel<binary_path, PackedLhsScalar, PackedRhsScalar,
    //                     DstScalar, TSpec>;

    ruy::TrMul(&binary_trmul_params, ruy_context);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
