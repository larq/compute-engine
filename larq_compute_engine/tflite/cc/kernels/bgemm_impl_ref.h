#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

using namespace tflite;
using namespace tflite::cpu_backend_gemm;

namespace compute_engine {
namespace tflite {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct BGemmImplRef {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    // TODO:
  }
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_REF_H_
