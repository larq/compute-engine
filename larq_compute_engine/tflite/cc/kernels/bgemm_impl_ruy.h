#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_

#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

template <typename Scalar, typename DataPointer>
void MakeRuyMatrix(const MatrixParams<Scalar>& params, DataPointer data_ptr,
                   ruy::Matrix<Scalar>* dst) {
  dst->layout.rows = params.rows;
  dst->layout.cols = params.cols;
  if (params.order == Order::kColMajor) {
    dst->layout.order = ruy::Order::kColMajor;
    dst->layout.stride = params.rows;
  } else {
    dst->layout.order = ruy::Order::kRowMajor;
    dst->layout.stride = params.cols;
  }
  // Note that ruy::Matrix::data is a ConstCheckingPtr, not a plain pointer.
  // It does care whether we assign to it a Scalar* or a const Scalar*.
  dst->data = data_ptr;
  dst->zero_point = params.zero_point;
}

template <typename GemmParamsType, typename RuySpecType>
void MakeRuySpec(const GemmParamsType& params, RuySpecType* ruy_spec) {
  // This validation has already been performed by the Gemm API entry point,
  // but it doesn't hurt to test specifically this again here, where it's
  // being used.
  ValidateGemmParams(params);

  ruy_spec->multiplier_fixedpoint = params.multiplier_fixedpoint;
  ruy_spec->multiplier_exponent = params.multiplier_exponent;
  ruy_spec->multiplier_fixedpoint_perchannel =
      params.multiplier_fixedpoint_perchannel;
  ruy_spec->multiplier_exponent_perchannel =
      params.multiplier_exponent_perchannel;
  ruy_spec->bias = params.bias;
  ruy_spec->clamp_min = params.clamp_min;
  ruy_spec->clamp_max = params.clamp_max;
}

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
    ruy::Matrix<LhsScalar> ruy_lhs;
    ruy::Matrix<RhsScalar> ruy_rhs;
    ruy::Matrix<DstScalar> ruy_dst;
    MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs);
    MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs);
    MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::BasicSpec<AccumScalar, DstScalar> ruy_spec;
    MakeRuySpec(params, &ruy_spec);

    // TODO: need to be modified to pickup the bgemm kernel
    // ruy::Mul<ruy::kAllPaths>(ruy_lhs, ruy_rhs, ruy_spec,
    // context->ruy_context(),
    //                          &ruy_dst);
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_RUY_H_
