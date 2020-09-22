#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_

#include <cstdint>
#include <type_traits>

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {
namespace kernel_4x2_portable {

/**
 * A 4x2 C++ micro-kernel for float or int8 output.
 */
template <typename DstScalar>
void RunKernel(const std::int32_t block_num_pixels,
               const std::int32_t conv_kernel_size,
               const std::int32_t channels_in, const std::int32_t channels_out,
               const bconv2d::OutputTransform<DstScalar>& output_transform,
               const TBitpacked* weights_ptr,
               const TBitpacked** indirection_buffer, DstScalar* output_ptr) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");

  ruy::profiler::ScopeLabel label("Indirect BGEMM block (4x2, portable)");

  TFLITE_DCHECK_GE(block_num_pixels, 1);
  TFLITE_DCHECK_LE(block_num_pixels, 2);
  TFLITE_DCHECK_GE(conv_kernel_size, 1);
  TFLITE_DCHECK_GE(channels_in, 1);
  TFLITE_DCHECK_GE(channels_out, 1);

  DstScalar* output_ptr_0 = output_ptr;
  DstScalar* output_ptr_1 = output_ptr + channels_out;

  // At the end of the output array we might get a block where the number of
  // pixels is less than 2, if the overall output size is not a multiple of 2.
  // When this happens we set the 'leftover' output pointer equal to the first
  // output pointer, so that there's no risk of writing beyond the array bounds.
  // At the end, when we write to the output array, we do it 'back to front' so
  // that the outputs for the first pixel are written last, which means that the
  // result will still be correct.
  if (block_num_pixels < 2) {
    output_ptr_1 = output_ptr_0;
  }

  std::int32_t c_out_index = 0;
  do {
    // Accumulators
    std::int32_t acc_00 = 0, acc_01 = 0;
    std::int32_t acc_10 = 0, acc_11 = 0;
    std::int32_t acc_20 = 0, acc_21 = 0;
    std::int32_t acc_30 = 0, acc_31 = 0;

    std::int32_t k_size_index = conv_kernel_size;
    do {
      const TBitpacked* activations_ptr_0 = indirection_buffer[0];
      const TBitpacked* activations_ptr_1 = indirection_buffer[1];
      indirection_buffer += 2;

      std::int32_t c_in_index = channels_in;
      do {
        const TBitpacked w_0 = weights_ptr[0];
        const TBitpacked w_1 = weights_ptr[1];
        const TBitpacked w_2 = weights_ptr[2];
        const TBitpacked w_3 = weights_ptr[3];
        weights_ptr += 4;

        const TBitpacked a_0 = *activations_ptr_0++;
        const TBitpacked a_1 = *activations_ptr_1++;

        acc_00 += xor_popcount(w_0, a_0);
        acc_10 += xor_popcount(w_1, a_0);
        acc_20 += xor_popcount(w_2, a_0);
        acc_30 += xor_popcount(w_3, a_0);
        acc_01 += xor_popcount(w_0, a_1);
        acc_11 += xor_popcount(w_1, a_1);
        acc_21 += xor_popcount(w_2, a_1);
        acc_31 += xor_popcount(w_3, a_1);
      } while (--c_in_index > 0);
    } while (--k_size_index > 0);

    if (channels_out - c_out_index >= 4) {
      output_ptr_1[0] = output_transform.Run(acc_01, c_out_index);
      output_ptr_1[1] = output_transform.Run(acc_11, c_out_index + 1);
      output_ptr_1[2] = output_transform.Run(acc_21, c_out_index + 2);
      output_ptr_1[3] = output_transform.Run(acc_31, c_out_index + 3);
      output_ptr_1 += 4;
      output_ptr_0[0] = output_transform.Run(acc_00, c_out_index);
      output_ptr_0[1] = output_transform.Run(acc_10, c_out_index + 1);
      output_ptr_0[2] = output_transform.Run(acc_20, c_out_index + 2);
      output_ptr_0[3] = output_transform.Run(acc_30, c_out_index + 3);
      output_ptr_0 += 4;

      indirection_buffer -= 2 * conv_kernel_size;
      c_out_index += 4;
    } else {
      if (channels_out - c_out_index >= 2) {
        output_ptr_1[0] = output_transform.Run(acc_01, c_out_index);
        output_ptr_1[1] = output_transform.Run(acc_11, c_out_index + 1);
        output_ptr_1 += 2;
        output_ptr_0[0] = output_transform.Run(acc_00, c_out_index);
        output_ptr_0[1] = output_transform.Run(acc_10, c_out_index + 1);
        output_ptr_0 += 2;

        acc_01 = acc_21;
        acc_00 = acc_20;
        c_out_index += 2;
      }
      if (channels_out - c_out_index >= 1) {
        output_ptr_1[0] = output_transform.Run(acc_01, c_out_index);
        output_ptr_0[0] = output_transform.Run(acc_00, c_out_index);
      }

      c_out_index = channels_out;
    }
  } while (c_out_index < channels_out);
}

/**
 * A 4x2 C++ micro-kernel for bitpacked output.
 */
template <>
void RunKernel<TBitpacked>(
    const std::int32_t block_num_pixels, const std::int32_t conv_kernel_size,
    const std::int32_t channels_in, const std::int32_t channels_out,
    const bconv2d::OutputTransform<TBitpacked>& output_transform,
    const TBitpacked* weights_ptr, const TBitpacked** indirection_buffer,
    TBitpacked* output_ptr) {
  ruy::profiler::ScopeLabel label("Indirect BGEMM block (4x2, portable)");

  TFLITE_DCHECK_GE(block_num_pixels, 1);
  TFLITE_DCHECK_LE(block_num_pixels, 2);
  TFLITE_DCHECK_GE(conv_kernel_size, 1);
  TFLITE_DCHECK_GE(channels_in, 1);
  TFLITE_DCHECK_GE(channels_out, 1);

  TBitpacked* output_ptr_0 = output_ptr;
  TBitpacked* output_ptr_1 =
      output_ptr + bitpacking::GetBitpackedSize(channels_out);

  // At the end of the output array we might get a block where the number of
  // pixels is less than 2, if the overall output size is not a multiple of 2.
  // When this happens we set the 'leftover' output pointer equal to the first
  // output pointer, so that there's no risk of writing beyond the array bounds.
  // At the end, when we write to the output array, we do it 'back to front' so
  // that the outputs for the first pixel are written last, which means that the
  // result will still be correct.
  if (block_num_pixels < 2) {
    output_ptr_1 = output_ptr_0;
  }

  // We will accumulate bits into these per-pixel columns and write a bitpacked
  // value when the columns are full.
  TBitpacked output_col_0 = 0, output_col_1 = 0;

  std::int32_t c_out_index = 0;
  do {
    // Accumulators
    std::int32_t acc_00 = 0, acc_01 = 0;
    std::int32_t acc_10 = 0, acc_11 = 0;
    std::int32_t acc_20 = 0, acc_21 = 0;
    std::int32_t acc_30 = 0, acc_31 = 0;

    std::int32_t k_size_index = conv_kernel_size;
    do {
      const TBitpacked* activations_ptr_0 = indirection_buffer[0];
      const TBitpacked* activations_ptr_1 = indirection_buffer[1];
      indirection_buffer += 2;

      std::int32_t c_in_index = channels_in;
      do {
        const TBitpacked w_0 = weights_ptr[0];
        const TBitpacked w_1 = weights_ptr[1];
        const TBitpacked w_2 = weights_ptr[2];
        const TBitpacked w_3 = weights_ptr[3];
        weights_ptr += 4;

        const TBitpacked a_0 = *activations_ptr_0++;
        const TBitpacked a_1 = *activations_ptr_1++;

        acc_00 += xor_popcount(w_0, a_0);
        acc_10 += xor_popcount(w_1, a_0);
        acc_20 += xor_popcount(w_2, a_0);
        acc_30 += xor_popcount(w_3, a_0);
        acc_01 += xor_popcount(w_0, a_1);
        acc_11 += xor_popcount(w_1, a_1);
        acc_21 += xor_popcount(w_2, a_1);
        acc_31 += xor_popcount(w_3, a_1);
      } while (--c_in_index > 0);
    } while (--k_size_index > 0);

    output_col_0 |= TBitpacked(output_transform.Run(acc_00, c_out_index))
                    << (c_out_index % bitpacking_bitwidth);
    output_col_0 |= TBitpacked(output_transform.Run(acc_10, c_out_index + 1))
                    << ((c_out_index + 1) % bitpacking_bitwidth);
    output_col_0 |= TBitpacked(output_transform.Run(acc_20, c_out_index + 2))
                    << ((c_out_index + 2) % bitpacking_bitwidth);
    output_col_0 |= TBitpacked(output_transform.Run(acc_30, c_out_index + 3))
                    << ((c_out_index + 3) % bitpacking_bitwidth);
    output_col_1 |= TBitpacked(output_transform.Run(acc_01, c_out_index))
                    << (c_out_index % bitpacking_bitwidth);
    output_col_1 |= TBitpacked(output_transform.Run(acc_11, c_out_index + 1))
                    << ((c_out_index + 1) % bitpacking_bitwidth);
    output_col_1 |= TBitpacked(output_transform.Run(acc_21, c_out_index + 2))
                    << ((c_out_index + 2) % bitpacking_bitwidth);
    output_col_1 |= TBitpacked(output_transform.Run(acc_31, c_out_index + 3))
                    << ((c_out_index + 3) % bitpacking_bitwidth);

    indirection_buffer -= 2 * conv_kernel_size;
    c_out_index += 4;

    // Write the bitpacked columns whenever they are full, or if we've computed
    // the last output column value.
    if (c_out_index % bitpacking_bitwidth == 0 || c_out_index >= channels_out) {
      // If this is a 'leftover output channel' block (because the number of
      // output channels isn't a multiple of four) then zero-out the extra bits.
      if (c_out_index % bitpacking_bitwidth != 0) {
        output_col_0 &=
            (TBitpacked(1) << (channels_out % bitpacking_bitwidth)) - 1;
        output_col_1 &=
            (TBitpacked(1) << (channels_out % bitpacking_bitwidth)) - 1;
      }

      *output_ptr_1++ = output_col_1;
      output_col_1 = 0;
      *output_ptr_0++ = output_col_0;
      output_col_0 = 0;
    }
  } while (c_out_index < channels_out);
}

}  // namespace kernel_4x2_portable
}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_
