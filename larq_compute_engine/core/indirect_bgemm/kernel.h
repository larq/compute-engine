
#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_

#include <cstdint>
#include <type_traits>

#include "larq_compute_engine/core/indirect_bgemm/kernel_4x2_portable.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_params.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

using compute_engine::tflite::bconv2d::TfLiteBConv2DParams;

template <typename DstScalar>
struct IndirectBGEMMKernel {
  using MicroKernelFunction = void(const std::int32_t, const std::int32_t,
                                   const std::int32_t, const std::int32_t,
                                   const bconv2d::OutputTransform<DstScalar>&,
                                   const TBitpacked*, const TBitpacked**,
                                   DstScalar*);
  MicroKernelFunction* micro_kernel_function;
  const std::int32_t block_size_output_channels;
  const std::int32_t block_size_pixels;
};

// This function allows us to select which kernel to use at runtime based on any
// parameter we choose: destination scalar; conv params; input/output shapes;
// even detected CPU features.
//     It is very important that this function is deterministic, as we rely on
// the fact that the same kernel is selected for each call to `Eval` (as long as
// the input shape doesn't change).
template <typename DstScalar>
inline IndirectBGEMMKernel<DstScalar> SelectRuntimeKernel(
    const TfLiteBConv2DParams* conv_params,
    const RuntimeShape& bitpacked_input_shape,
    const RuntimeShape& output_shape) {
  // For now there is only one kernel available.
  return IndirectBGEMMKernel<DstScalar>{
      &kernel_4x2_portable::RunKernel<DstScalar>, 4, 2};
}

template <typename DstScalar>
void RunKernel(const IndirectBGEMMKernel<DstScalar>& kernel,
               const std::int32_t conv_kernel_size,
               const std::int32_t bitpacked_input_channels,
               const std::int32_t output_size,
               const std::int32_t output_channels,
               const bconv2d::OutputTransform<DstScalar>& output_transform,
               const TBitpacked* packed_weights_ptr,
               const TBitpacked** indirection_buffer, DstScalar* output_ptr) {
  // TODO: implement multithreading here.
  for (std::int32_t pixel_start = 0; pixel_start < output_size;
       pixel_start += kernel.block_size_pixels) {
    const std::int32_t output_stride =
        std::is_same<DstScalar, TBitpacked>::value
            ? bitpacking::GetBitpackedSize(output_channels)
            : output_channels;
    kernel.micro_kernel_function(
        std::min(output_size - pixel_start, kernel.block_size_pixels),
        conv_kernel_size, bitpacked_input_channels, output_channels,
        output_transform, packed_weights_ptr,
        indirection_buffer + pixel_start * conv_kernel_size,
        output_ptr + pixel_start * output_stride);
  }
}

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_
