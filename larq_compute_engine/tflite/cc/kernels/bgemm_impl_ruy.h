#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

#include "tensorflow/lite/experimental/ruy/ruy_advanced.h"

#include "larq_compute_engine/cc/core/bgemm_functor.h"

// #include <iostream>

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {

namespace ce = compute_engine;

namespace tflite {

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct BinaryKernel {};

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct BinaryKernel<ruy::Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar, Spec> {
  using AccumScalar = typename Spec::AccumScalar;
  using LhsLayout = typename Spec::StandardCppKernelLhsLayout;
  using RhsLayout = typename Spec::StandardCppKernelRhsLayout;
  explicit BinaryKernel(ruy::Tuning) {}
  void Run(const ruy::PackedMatrix<LhsScalar>& lhs,
           const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec, int start_row,
           int start_col, int end_row, int end_col,
           ruy::Matrix<DstScalar>* dst) const {
    // See the comment in RunKernelTyped. end_row may be larger than
    // dst->layout.rows. It's the responsibility of the kernel to avoid
    // overrunning dst boundaries, which we do here by computing
    // clamped_end_row.
    int clamped_end_row = std::min(end_row, dst->layout.rows);
    int clamped_end_col = std::min(end_col, dst->layout.cols);
    RUY_DCHECK_LE(0, start_row);
    RUY_DCHECK_LE(start_row, clamped_end_row);
    RUY_DCHECK_LE(clamped_end_row, dst->layout.rows);
    RUY_DCHECK_LE(clamped_end_row, end_row);
    RUY_DCHECK_LE(end_row - clamped_end_row, LhsLayout::kCols);
    RUY_DCHECK_LE(0, start_col);
    RUY_DCHECK_LE(start_col, clamped_end_col);
    RUY_DCHECK_LE(clamped_end_col, dst->layout.cols);
    RUY_DCHECK_LE(clamped_end_col, end_col);
    RUY_DCHECK_LE(end_col - clamped_end_col, RhsLayout::kCols);
    gemmlowp::ScopedProfilingLabel label("Binary Kernel (Standard Cpp)");
    // std::cout << "Binary Kernel (Standard Cpp)" << std::endl;
    const int depth = lhs.layout.rows;
    // std::cout << "depth: " << depth << std::endl;
    using TBitpacked = std::uint8_t;
    for (int i = start_row; i < clamped_end_row; i++) {
      for (int j = start_col; j < clamped_end_col; j++) {
        using AccumScalar = typename Spec::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          // TODO: implicit casting should be replaced with passing templates
          // after elements are passed as TBitpacked instead of float
          TBitpacked lhs_val = Element(lhs, k, i);
          TBitpacked rhs_val = Element(rhs, k, j);
          accum += ce::core::compute_binary_inner_prod<TBitpacked, AccumScalar>(lhs_val, rhs_val);
          // std::cout << "ACCUM: " << accum << std::endl;
        }
        if (spec.bias) {
          accum += spec.bias[i];
        }
        // if (lhs.zero_point) {
        //   accum -= lhs.zero_point * rhs.sums[j];
        // }
        // if (rhs.zero_point) {
        //   accum -= rhs.zero_point * lhs.sums[i];
        // }
        // if (lhs.zero_point && rhs.zero_point) {
        //   accum += lhs.zero_point * rhs.zero_point * depth;
        // }
        // ApplyMultiplier(spec, i, &accum);
        // accum += dst->zero_point;
        // accum = std::min<AccumScalar>(accum, spec.clamp_max);
        // accum = std::max<AccumScalar>(accum, spec.clamp_min);
        *ElementPtr(dst, i, j) = static_cast<DstScalar>(accum) - spec.multiplier_exponent;
      }
    }
  }
};

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunBinaryKernelTyped(ruy::Tuning tuning, const ruy::PackedMatrix<LhsScalar>& lhs,
                          const ruy::PackedMatrix<RhsScalar>& rhs, const Spec& spec,
                          int start_row, int start_col, int end_row, int end_col,
                          ruy::Matrix<DstScalar>* dst) {
  // std::cout << "RUN BINARY KERNEL TYPED\n";
  using BKernel = BinaryKernel<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>;
  BKernel kernel(tuning);
  using LhsLayout = typename BKernel::LhsLayout;
  using RhsLayout = typename BKernel::RhsLayout;
  // end_row and end_col may be larger than dst dimensions.
  // that is because kernels write directly to the destination matrix, whose
  // dimensions may not be a multiple of the kernel dimensions, and we try to
  // keep this annoyance localized as an implementation detail in kernels,
  // by allowing to pass rounded-up values down as far as possible.
  // These assertions encode the contract.
  RUY_DCHECK_LE(0, start_row);
  RUY_DCHECK_LE(start_row, end_row);
  RUY_DCHECK_LT(end_row, dst->layout.rows + LhsLayout::kCols);
  RUY_DCHECK_EQ((end_row - start_row) % LhsLayout::kCols, 0);
  RUY_DCHECK_LE(0, start_col);
  RUY_DCHECK_LE(start_col, end_col);
  RUY_DCHECK_LT(end_col, dst->layout.cols + RhsLayout::kCols);
  RUY_DCHECK_EQ((end_col - start_col) % RhsLayout::kCols, 0);
#if RUY_OPT_ENABLED(RUY_OPT_FAT_KERNEL)
  // std::cout << "RUNING fat kernel: ROW(" << start_row << " , " << end_row << ")\n";
  // std::cout << "                 : COL(" << start_col << " , " << end_col << ")\n";
  kernel.Run(lhs, rhs, spec, start_row, start_col, end_row, end_col, dst);
#else
  for (int col = start_col; col < end_col; col += RhsLayout::kCols) {
    int block_end_col = std::min(col + RhsLayout::kCols, end_col);
    for (int row = start_row; row < end_row; row += LhsLayout::kCols) {
      int block_end_row = std::min(row + LhsLayout::kCols, end_row);
      // std::cout << "RUNING kernel: ROW(" << row << " , " << block_end_row << ")\n";
      // std::cout << "             : COL(" << col << " , " << block_end_col << ")\n";
      kernel.Run(lhs, rhs, spec, row, col, block_end_row, block_end_col, dst);
    }
  }
#endif
}

template <ruy::Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunBinaryKernel(ruy::Tuning tuning, const ruy::SidePair<ruy::PMatrix>& src, void* spec,
                     const ruy::SidePair<int>& start, const ruy::SidePair<int>& end,
                     ruy::DMatrix* dst) {
  // std::cout << "RUN BINARY KERNEL" << std::endl;
  ruy::Matrix<DstScalar> mdst = ruy::ToMatrix<DstScalar>(*dst);
  RunBinaryKernelTyped<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>(
      tuning, ruy::ToPackedMatrix<LhsScalar>(src[ruy::Side::kLhs]),
      ruy::ToPackedMatrix<RhsScalar>(src[ruy::Side::kRhs]),
      *static_cast<const Spec*>(spec), start[ruy::Side::kLhs], start[ruy::Side::kRhs],
      end[ruy::Side::kLhs], end[ruy::Side::kRhs], &mdst);
}

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
    auto ruy_context = context->ruy_context();

    // convert to float
    // using T = std::uint8_t;
    using T = float;
    using TAccum = std::int32_t;
    using TSpec = ruy::BasicSpec<TAccum, DstScalar>;

    // TODO: conversion to float is temporary to avoid executing of RUY 8-bit quantized GEMM
    // until we manually replace the kernel function pointers with our binary kernels
    std::vector<LhsScalar> lhs_data_int_vec(lhs_data, lhs_data + sizeof lhs_data / sizeof lhs_data[0]);
    std::vector<RhsScalar> rhs_data_int_vec(rhs_data, rhs_data + sizeof rhs_data / sizeof rhs_data[0]);
    std::vector<T> ruy_lhs_fl (lhs_data_int_vec.begin(), lhs_data_int_vec.end());
    std::vector<T> ruy_rhs_fl (rhs_data_int_vec.begin(), rhs_data_int_vec.end());
    const T* lhs_data_fl = ruy_lhs_fl.data();
    const T* rhs_data_fl = ruy_rhs_fl.data();

    // Set up the matrix layouts and spec.
    ruy::Matrix<T> lhs;
    ruy::MakeSimpleLayout(lhs_params.rows, lhs_params.cols, ruy::Order::kRowMajor, &lhs.layout);
    ruy::Matrix<T> rhs;
    ruy::MakeSimpleLayout(rhs_params.rows, rhs_params.cols, ruy::Order::kColMajor, &rhs.layout);
    ruy::Matrix<DstScalar> dst;
    ruy::MakeSimpleLayout(dst_params.rows, dst_params.cols, ruy::Order::kColMajor, &dst.layout);

    TSpec spec;

    // The allocator is used to allocate memory for pre-packed matrices
    // TODO: needs aligned allocator for NEON SIMD?
    SimpleAllocator allocator;
    auto alloc_fn = [&allocator](std::size_t num_bytes) -> void* {
                      return allocator.AllocateBytes(num_bytes);
                    };

    ruy::PrepackedMatrix prepacked_lhs;
    lhs.data = lhs_data_fl;

    ruy::PrepackedMatrix prepacked_rhs;
    rhs.data = rhs_data_fl;

    ruy::PrePackForMul<ruy::kAllPaths>(lhs, rhs, spec, ruy_context, &dst,
                                       &prepacked_lhs, &prepacked_rhs,
                                       alloc_fn);

    lhs.data = nullptr;
    rhs.data = nullptr;
    dst.data = dst_data;

    // avoid the reference path for production code
    ruy::Path the_path = ruy_context->GetPathToTake<ruy::kAllPaths>();
    RUY_CHECK_NE(the_path, ruy::Path::kReference);



    // Here, we abuse the
    // 'multiplier_exponent' which is used only for non-floating-point
    // cases to pass the bitpadding correction value (int) to bgemm kernel
    spec.multiplier_exponent = params.multiplier_exponent;

    // In Ruy, TrMul is computed instead of Mul, therefore the lhs needs to be transposed.
    // Transpose function is cheap since it does not shuffle data around and only
    // changes the matrix layout.
    ruy::Matrix<T> transposed_lhs(lhs);
    ruy::Transpose(&transposed_lhs);

    // Based on the Path, kernel function pointers are set in TrMulParams
    constexpr ruy::Path TrMulCompiledPaths = ruy::kAllPaths & ~ruy::Path::kReference;
    ruy::TrMulParams trmul_params;
    ruy::CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, ruy_context, &dst,
                                               the_path, &trmul_params);

    // set the pre-packed params
    trmul_params.packed[ruy::Side::kLhs].data = prepacked_lhs.data;
    trmul_params.packed[ruy::Side::kLhs].sums = prepacked_lhs.sums;
    trmul_params.is_prepacked[ruy::Side::kLhs] = true;

    trmul_params.packed[ruy::Side::kRhs].data = prepacked_rhs.data;
    trmul_params.packed[ruy::Side::kRhs].sums = prepacked_rhs.sums;
    trmul_params.is_prepacked[ruy::Side::kRhs] = true;

    // TODO: redirect the kernel function pointer to the binary kernel
    // of the corresponding path (architecture) determind on the compiled time
    using PackedLhsScalar = ruy::PackedType<ruy::Path::kStandardCpp, T>;
    using PackedRhsScalar = ruy::PackedType<ruy::Path::kStandardCpp, T>;
    trmul_params.run_kernel = &RunBinaryKernel<ruy::Path::kStandardCpp, PackedLhsScalar, PackedRhsScalar, DstScalar, TSpec>;

    ruy::TrMul(&trmul_params, ruy_context);

    // reseting to original values after using the prepacked values
    lhs.data = lhs_data_fl;
    rhs.data = rhs_data_fl;

    // Print out the results.
    // std::cout << "LHS:\n" << lhs;
    // std::cout << "RHS:\n" << rhs;
    // std::cout << "Result:\n" << dst << "\n";

  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
