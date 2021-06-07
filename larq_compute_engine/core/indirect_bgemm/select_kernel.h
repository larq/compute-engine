#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_SELECT_KERNEL_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_SELECT_KERNEL_H_

#include <cstdint>
#include <memory>
#include <type_traits>

#include "larq_compute_engine/core/indirect_bgemm/kernel.h"

#ifdef __aarch64__
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x1_aarch64.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x2_aarch64.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_8x4x4_aarch64.h"
#endif
#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel_4x2_portable.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

// These functions allow us to select which kernel to use at runtime based on
// any parameter we choose: destination scalar; conv params; input/output
// shapes; even detected CPU features.

// Select a kernel for float or int8 output.
template <typename DstScalar>
inline std::unique_ptr<Kernel> SelectRuntimeKernel(
    const bconv2d::BConv2DParams* bconv2d_params,
    const tflite::RuntimeShape& bitpacked_input_shape,
    const tflite::RuntimeShape& output_shape,
    const bconv2d::OutputTransform<DstScalar>& output_transform) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");

#ifdef __aarch64__
  // There are optimised assembly kernels for float and int8 output on Aarch64.
  // They all use int16 accumulators. A different kernel is selected depending
  // on whether the bitpacked number of input channels is a multiple of 4, 2, or
  // 1, and whether that multiple is 1 or more than 1.

  const auto max_accumulator_value = bitpacking_bitwidth *
                                     bitpacked_input_shape.FlatSize() /
                                     bconv2d_params->groups;
  const bool fits_in_uint16_accumulators =
      max_accumulator_value < std::numeric_limits<std::uint16_t>::max();

  if (fits_in_uint16_accumulators) {
    const auto input_depth_per_group =
        bitpacked_input_shape.Dims(3) / bconv2d_params->groups;

    if (input_depth_per_group % 4 == 0) {
      if (input_depth_per_group > 4) {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x4Aarch64<DstScalar, true, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        } else {
          return std::make_unique<Kernel8x4x4Aarch64<DstScalar, true, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      } else {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x4Aarch64<DstScalar, false, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        } else {
          return std::make_unique<Kernel8x4x4Aarch64<DstScalar, false, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      }
    } else if (input_depth_per_group % 2 == 0) {
      if (input_depth_per_group > 2) {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x2Aarch64<DstScalar, true, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        } else {
          return std::make_unique<Kernel8x4x2Aarch64<DstScalar, true, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      } else {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x2Aarch64<DstScalar, false, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        } else {
          return std::make_unique<Kernel8x4x2Aarch64<DstScalar, false, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      }
    } else {
      if (input_depth_per_group > 1) {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x1Aarch64<DstScalar, true, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);

        } else {
          return std::make_unique<Kernel8x4x1Aarch64<DstScalar, true, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      } else {
        if (bconv2d_params->groups > 1) {
          return std::make_unique<Kernel8x4x1Aarch64<DstScalar, false, true>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);

        } else {
          return std::make_unique<Kernel8x4x1Aarch64<DstScalar, false, false>>(
              bconv2d_params, bitpacked_input_shape, output_shape,
              output_transform);
        }
      }
    }
  }
#endif

  // Fallback C++ kernel
  return std::make_unique<Kernel4x2Portable<DstScalar>>(
      bconv2d_params, bitpacked_input_shape, output_shape, output_transform);
}

// A specialisation: select a kernel for bitpacked output.
template <>
inline std::unique_ptr<Kernel> SelectRuntimeKernel<TBitpacked>(
    const bconv2d::BConv2DParams* bconv2d_params,
    const tflite::RuntimeShape& bitpacked_input_shape,
    const tflite::RuntimeShape& output_shape,
    const bconv2d::OutputTransform<TBitpacked>& output_transform) {
  // Only the C++ kernel currently supports bitpacked output.
  return std::make_unique<Kernel4x2Portable<TBitpacked>>(
      bconv2d_params, bitpacked_input_shape, output_shape, output_transform);
}

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_SELECT_KERNEL_H_
