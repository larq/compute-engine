#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_TRMUL_PARAMS_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_TRMUL_PARAMS_H_

#include "bgemm_kernels_ruy.h"
#include "ruy/dispatch.h"
#include "ruy/mul_params.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
#include "ruy/trmul_params.h"

using namespace ruy;

namespace compute_engine {
namespace tflite {

// This file is entirely copied from TF lite RUY codebase. The only difference
// is that we populate the TrMulParams with our bgemm kernel.

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void PopulateBinaryTrMulParams(ruy::TrMulParams* params) {
  static_assert((ThePath & ruy::Path::kReference) == ruy::Path::kNone,
                "Path::kReference should not do TrMul");
  // The optimized code paths don't handle the full generality of Ruy's API.
  // Fall back to Path::kStandardCpp if necessary.
  bool fallback_to_standard_cpp = false;
  if (ThePath != ruy::Path::kStandardCpp) {
    // The optimized code paths currently only handle the case of all matrices
    // being column major.
    if (!ruy::IsColMajorTrMul(*params)) {
      fallback_to_standard_cpp = true;
    }
  }

  if (fallback_to_standard_cpp) {
    PopulateBinaryTrMulParams<ruy::Path::kStandardCpp, LhsScalar, RhsScalar,
                              DstScalar, MulParamsType>(params);
    return;
  }

  // TODO: why packed values type is different than LhsScalar
  using PackedLhsScalar = ruy::PackedType<ThePath, LhsScalar>;
  using PackedRhsScalar = ruy::PackedType<ThePath, RhsScalar>;
  using Kernel = BgemmKernel<ThePath, PackedLhsScalar, PackedRhsScalar,
                             DstScalar, MulParamsType>;
  using LhsKernelLayout = typename Kernel::LhsLayout;
  using RhsKernelLayout = typename Kernel::RhsLayout;

  params->path = ThePath;

  params->local_data_cache_size = MulParamsType::local_data_cache_size();
  params->shared_data_cache_size = MulParamsType::shared_data_cache_size();

  CreatePackedMatrix<LhsScalar, PackedLhsScalar>(
      Side::kLhs, ToKernelLayout<LhsKernelLayout>(), params);
  CreatePackedMatrix<RhsScalar, PackedRhsScalar>(
      Side::kRhs, ToKernelLayout<RhsKernelLayout>(), params);
  params->run_pack[Side::kLhs] =
      &RunPack<ThePath, LhsKernelLayout, LhsScalar, PackedLhsScalar>;
  params->run_pack[Side::kRhs] =
      &RunPack<ThePath, RhsKernelLayout, RhsScalar, PackedRhsScalar>;
  params->run_kernel =
      &RunBgemmKernel<ThePath, PackedLhsScalar, PackedRhsScalar, DstScalar,
                      MulParamsType>;
  return;
}

// PopulateTrMulParamsAllCompiledPaths calls into one of multiple
// instantiations of PopulateTrMulParams. For each bit that is set in
// CompiledPaths, it statically instantiates PopulateTrMulParams with a Path
// corresponding to that single bit. The call to PopulateTrMulParams is
// guarded by a runtime check that it is in fact the dynamically selected path.
//
// PopulateTrMulParamsAllCompiledPaths is implemented with template
// metaprogramming by mutual recursion between PathSearchCountdown and
// PathSearchCompiledPaths.
//
// PopulateTrMulParamsAllCompiledPaths is logically implementing the following
// computation:
//
// template <Path CompiledPaths>
// void PopulateTrMulParamsAllCompiledPaths(Path the_path,
//                                            TrMulParams* params) {
//   for (int bit = 8 * sizeof(Path) - 1; bit != -1; bit--) { // [1]
//     Path current_path = static_cast<Path>(1 << bit);
//     if ((CompiledPaths & current_path) != Path::kNone) { // [2]
//       if (current_path == the_path) { // [3]
//         PopulateTrMulParams<current_path, ...>(the_path, params);
//         return;
//       }
//     }
//   }
// }
//
//
//
// [1] - Done by the main definition of PathSearchCountdown. The `bit--` is
// done in the recursion of PathSearchOnlyCompiledPaths.
// [2] - Done by PathSearchOnlyCompiledPaths's partial template
// specialization on InCompiledPaths. This is the check which necessitates
// doing the whole computation at C++ compile time.
// [3] - Done by the `if` in the main definition of
// PathSearchOnlyCompiledPaths.
//
// The template metaprogramming is necessary because:
// - In `PopulateTrMulParams<current_path, ...>`, current_path must be a C++
// compile-time constant.
// - PopulateTrMulParamsAllCompiledPaths must not instantiate
// inner loops for paths that are not in CompiledPaths, since that can result in
// bogus instantiations which cause a compile time failure.
template <ruy::Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchCountdown;

template <ruy::Path CompiledPaths, bool InCompiledPaths, int BitNumber,
          typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct PathSearchOnlyCompiledPaths {
  static constexpr ruy::Path kCurrentPath =
      static_cast<ruy::Path>(1 << BitNumber);
  static void Search(ruy::Path the_path, ruy::TrMulParams* params) {
    if (kCurrentPath == the_path) {
      PopulateBinaryTrMulParams<kCurrentPath, LhsScalar, RhsScalar, DstScalar,
                                MulParamsType>(params);
      return;
    }
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        DstScalar, MulParamsType>::Search(the_path, params);
  }
};

// Skip this iteration if CompiledPaths doesn't contain the specified path.
template <ruy::Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchOnlyCompiledPaths<CompiledPaths, false, BitNumber, LhsScalar,
                                   RhsScalar, DstScalar, MulParamsType> {
  static void Search(ruy::Path the_path, ruy::TrMulParams* params) {
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        DstScalar, MulParamsType>::Search(the_path, params);
  }
};

template <ruy::Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchCountdown {
  static constexpr ruy::Path kCurrentPath =
      static_cast<ruy::Path>(1 << BitNumber);
  static void Search(ruy::Path the_path, ruy::TrMulParams* params) {
    PathSearchOnlyCompiledPaths<
        CompiledPaths, (CompiledPaths & kCurrentPath) != ruy::Path::kNone,
        BitNumber, LhsScalar, RhsScalar, DstScalar,
        MulParamsType>::Search(the_path, params);
  }
};

// Termination of the countdown. If the counter reaches -1, then we haven't
// found the specified path.
template <ruy::Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
struct PathSearchCountdown<CompiledPaths, -1, LhsScalar, RhsScalar, DstScalar,
                           MulParamsType> {
  static void Search(ruy::Path the_path, ruy::TrMulParams* params) {
    RUY_DCHECK(false);
  }
};

template <ruy::Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void PopulateBinaryTrMulParamsAllCompiledPaths(ruy::Path the_path,
                                               ruy::TrMulParams* params) {
  return PathSearchCountdown<CompiledPaths, 8 * sizeof(ruy::Path) - 1,
                             LhsScalar, RhsScalar, DstScalar,
                             MulParamsType>::Search(the_path, params);
}

template <ruy::Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void CreateBinaryTrMulParams(const Matrix<LhsScalar>& lhs,
                             const Matrix<RhsScalar>& rhs,
                             const MulParamsType& mul_params, Context* context,
                             Matrix<DstScalar>* dst, Path the_path,
                             TrMulParams* params) {
  // Fill in the fields we already know.
  params->src[Side::kLhs] = ToDMatrix(lhs);
  params->src[Side::kRhs] = ToDMatrix(rhs);
  params->dst = ToDMatrix(*dst);
  params->mul_params = ToVoidPtr(&mul_params);

  // Create inner loops and packed matrices based on the Path.
  PopulateBinaryTrMulParamsAllCompiledPaths<CompiledPaths, LhsScalar, RhsScalar,
                                            DstScalar, MulParamsType>(the_path,
                                                                      params);
}

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_TRMUL_PARAMS_H_
