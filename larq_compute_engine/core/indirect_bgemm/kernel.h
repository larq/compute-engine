
#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_H_

#include <cstdint>
#include <type_traits>

#ifdef __aarch64__
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x1_aarch64.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x2_aarch64.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x4_aarch64.h"
#endif
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_4x2_portable.h"
#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

using namespace tflite;

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

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
  const std::int32_t block_size_depth;
};

// This function allows us to select which kernel to use at runtime based on any
// parameter we choose: destination scalar; conv params; input/output shapes;
// even detected CPU features.
//     It is very important that this function is deterministic, as we rely on
// the fact that the same kernel is selected for each call to `Eval` (as long as
// the input shape doesn't change).
template <typename DstScalar>
inline IndirectBGEMMKernel<DstScalar> SelectRuntimeKernel(
    const bconv2d::BConv2DParams* bconv2d_params,
    const RuntimeShape& bitpacked_input_shape,
    const RuntimeShape& output_shape) {
#ifdef __aarch64__
  // There are optimised assembly kernels for float and int8 output on Aarch64.
  // They all use int16 accumulators. A different kernel is selected depending
  // on whether the bitpacked number of input channels is a multiple of 4, 2, or
  // 1, and whether that multiple is 1 or more than 1.

  constexpr bool is_float_or_int8 = std::is_same<DstScalar, float>::value ||
                                    std::is_same<DstScalar, std::int8_t>::value;
  const int max_accumulator_value = bconv2d_params->filter_height *
                                    bconv2d_params->filter_width *
                                    bconv2d_params->channels_in;
  const bool fits_in_int16_accumulators = max_accumulator_value + 512 < 1 << 16;

  if (is_float_or_int8 && fits_in_int16_accumulators) {
    // This weirdness is required because DstScalar could be `TBitpacked`, but
    // in many cases RunKernel<TBitpacked, ...> isn't defined. As we can't use
    // `if constexpr` (C++17), this causes a compile error despite the fact that
    // we know that within this if block DstScalar must be float or int8.
    using DS =
        typename std::conditional<is_float_or_int8, DstScalar, float>::type;
    using KFn = typename IndirectBGEMMKernel<DstScalar>::MicroKernelFunction;

    if (bitpacked_input_shape.Dims(3) % 4 == 0) {
      if (bitpacked_input_shape.Dims(3) > 4) {
        return {(KFn*)&kernel_8x4x4_aarch64::RunKernel<DS, true>, 8, 4, 4};
      } else {
        return {(KFn*)&kernel_8x4x4_aarch64::RunKernel<DS, false>, 8, 4, 4};
      }
    } else if (bitpacked_input_shape.Dims(3) % 2 == 0) {
      if (bitpacked_input_shape.Dims(3) > 2) {
        return {(KFn*)&kernel_8x4x2_aarch64::RunKernel<DS, true>, 8, 4, 2};
      } else {
        return {(KFn*)&kernel_8x4x2_aarch64::RunKernel<DS, false>, 8, 4, 2};
      }
    } else {
      if (bitpacked_input_shape.Dims(3) > 1) {
        return {(KFn*)&kernel_8x4x1_aarch64::RunKernel<DS, true>, 8, 4, 1};
      } else {
        return {(KFn*)&kernel_8x4x1_aarch64::RunKernel<DS, false>, 8, 4, 1};
      }
    }
  }
#endif

  // Fallback C++ kernel
  return {&kernel_4x2_portable::RunKernel<DstScalar>, 4, 2, 1};
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
