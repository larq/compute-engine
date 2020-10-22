#ifndef COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_
#define COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_

#include <cstdint>
#include <type_traits>

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel.h"
#include "larq_compute_engine/core/types.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace indirect_bgemm {

/**
 * A 4x2 C++ micro-kernel for float or int8 output.
 */
template <typename DstScalar>
struct Kernel4x2Portable : Kernel {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");

  const bconv2d::OutputTransform<DstScalar> output_transform;

  Kernel4x2Portable(const bconv2d::BConv2DParams* bconv2d_params,
                    const RuntimeShape& bitpacked_input_shape,
                    const RuntimeShape& output_shape,
                    const bconv2d::OutputTransform<DstScalar>& output_transform)
      : Kernel(4, 2, 1, bconv2d_params, bitpacked_input_shape, output_shape),
        output_transform(output_transform) {}

  void Run(const std::int32_t pixel_start, const std::int32_t pixel_end,
           void* output_ptr) const override {
    ruy::profiler::ScopeLabel label("Indirect BGEMM block (4x2, portable)");

    TFLITE_DCHECK_GE(this->input_depth, 1);
    TFLITE_DCHECK_GE(this->output_channels, 1);
    TFLITE_DCHECK_GE(this->filter_size, 1);
    TFLITE_DCHECK_GE(this->groups, 1);
    TFLITE_DCHECK_EQ(this->input_depth % this->groups, 0);
    TFLITE_DCHECK_EQ(this->output_channels % this->groups, 0);

    const std::int32_t input_depth_per_group = this->input_depth / this->groups;
    const std::int32_t output_channels_per_group =
        this->output_channels / this->groups;

    for (std::int32_t p_index = pixel_start; p_index < pixel_end;
         p_index += 2) {
      const TBitpacked* weights_ptr = this->packed_weights.data();
      const TBitpacked* const* indirection_ptr =
          this->indirection_buffer.data() + p_index * this->filter_size;
      DstScalar* output_ptr_0 = reinterpret_cast<DstScalar*>(output_ptr) +
                                p_index * this->output_channels;
      DstScalar* output_ptr_1 = output_ptr_0 + this->output_channels;

      // At the end of the output array we might get a block where the number of
      // pixels is less than 2, if the overall output size is not a multiple
      // of 2. When this happens we set the 'leftover' output pointer equal to
      // the first output pointer, so that there's no risk of writing beyond the
      // array bounds. At the end, when we write to the output array, we do it
      // 'back to front' so that the outputs for the first pixel are written
      // last, which means that the result will still be correct.
      if (pixel_end - p_index < 2) {
        output_ptr_1 = output_ptr_0;
      }

      std::int32_t input_depth_offset = 0;
      std::int32_t group_end_output_channel = output_channels_per_group;

      std::int32_t c_out_index = 0;
      do {
        // Accumulators
        std::int32_t acc_00 = 0, acc_01 = 0;
        std::int32_t acc_10 = 0, acc_11 = 0;
        std::int32_t acc_20 = 0, acc_21 = 0;
        std::int32_t acc_30 = 0, acc_31 = 0;

        std::int32_t f_size_index = this->filter_size;
        do {
          const TBitpacked* activations_ptr_0 =
              indirection_ptr[0] + input_depth_offset;
          const TBitpacked* activations_ptr_1 =
              indirection_ptr[1] + input_depth_offset;
          indirection_ptr += 2;

          std::int32_t depth_index = input_depth_per_group;
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
          } while (--depth_index > 0);
        } while (--f_size_index > 0);

        if (LCE_LIKELY(group_end_output_channel - c_out_index >= 4)) {
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

          c_out_index += 4;
        } else {
          if (group_end_output_channel - c_out_index >= 2) {
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
          if (group_end_output_channel - c_out_index >= 1) {
            output_ptr_1[0] = output_transform.Run(acc_01, c_out_index);
            output_ptr_1 += 1;
            output_ptr_0[0] = output_transform.Run(acc_00, c_out_index);
            output_ptr_0 += 1;

            c_out_index += 1;
          }
        }

        indirection_ptr -= 2 * this->filter_size;

        if (c_out_index == group_end_output_channel) {
          input_depth_offset += input_depth_per_group;
          group_end_output_channel += output_channels_per_group;
        }
      } while (c_out_index < this->output_channels);
    }
  }
};

/**
 * A 4x2 C++ micro-kernel for bitpacked output.
 */
template <>
struct Kernel4x2Portable<TBitpacked> : Kernel {
  const bconv2d::OutputTransform<TBitpacked> output_transform;

  Kernel4x2Portable(
      const bconv2d::BConv2DParams* bconv2d_params,
      const RuntimeShape& bitpacked_input_shape,
      const RuntimeShape& output_shape,
      const bconv2d::OutputTransform<TBitpacked>& output_transform)
      : Kernel(4, 2, 1, bconv2d_params, bitpacked_input_shape, output_shape),
        output_transform(output_transform) {}

  void Run(const std::int32_t pixel_start, const std::int32_t pixel_end,
           void* output_ptr) const override {
    ruy::profiler::ScopeLabel label("Indirect BGEMM block (4x2, portable)");

    TFLITE_DCHECK_GE(this->input_depth, 1);
    TFLITE_DCHECK_GE(this->output_channels, 1);
    TFLITE_DCHECK_GE(this->filter_size, 1);
    TFLITE_DCHECK_GE(this->groups, 1);
    TFLITE_DCHECK_EQ(this->input_depth % this->groups, 0);
    TFLITE_DCHECK_EQ(this->output_channels % this->groups, 0);

    const std::int32_t input_depth_per_group = this->input_depth / this->groups;
    const std::int32_t output_channels_per_group =
        this->output_channels / this->groups;

    for (std::int32_t p_index = pixel_start; p_index < pixel_end;
         p_index += 2) {
      const TBitpacked* weights_ptr = this->packed_weights.data();
      const TBitpacked* const* indirection_ptr =
          this->indirection_buffer.data() + p_index * this->filter_size;
      TBitpacked* output_ptr_0 =
          reinterpret_cast<TBitpacked*>(output_ptr) +
          p_index * bitpacking::GetBitpackedSize(this->output_channels);
      TBitpacked* output_ptr_1 =
          output_ptr_0 + bitpacking::GetBitpackedSize(this->output_channels);

      // At the end of the output array we might get a block where the number of
      // pixels is less than 2, if the overall output size is not a multiple
      // of 2. When this happens we set the 'leftover' output pointer equal to
      // the first output pointer, so that there's no risk of writing beyond the
      // array bounds. At the end, when we write to the output array, we do it
      // 'back to front' so that the outputs for the first pixel are written
      // last, which means that the result will still be correct.
      if (pixel_end - p_index < 2) {
        output_ptr_1 = output_ptr_0;
      }

      // We will accumulate bits into these per-pixel columns and write a
      // bitpacked value when the columns are full.
      TBitpacked output_col_0 = 0, output_col_1 = 0;

      std::int32_t input_depth_offset = 0;
      std::int32_t group_end_output_channel = output_channels_per_group;

      std::int32_t c_out_index = 0;
      do {
        // Accumulators
        std::int32_t acc_00 = 0, acc_01 = 0;
        std::int32_t acc_10 = 0, acc_11 = 0;
        std::int32_t acc_20 = 0, acc_21 = 0;
        std::int32_t acc_30 = 0, acc_31 = 0;

        std::int32_t f_size_index = filter_size;
        do {
          const TBitpacked* activations_ptr_0 =
              indirection_ptr[0] + input_depth_offset;
          const TBitpacked* activations_ptr_1 =
              indirection_ptr[1] + input_depth_offset;
          indirection_ptr += 2;

          std::int32_t depth_index = input_depth_per_group;
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
          } while (--depth_index > 0);
        } while (--f_size_index > 0);

        // Correctness of the following section relies on the bitpacking
        // bitwidth being 32.
        static_assert(bitpacking_bitwidth == 32, "");

        const int base_output_index = c_out_index % 16;
        output_col_0 |= TBitpacked(output_transform.Run(acc_00, c_out_index))
                        << base_output_index;
        output_col_0 |=
            TBitpacked(output_transform.Run(acc_10, c_out_index + 1))
            << (base_output_index + 1);
        output_col_0 |=
            TBitpacked(output_transform.Run(acc_20, c_out_index + 2))
            << (base_output_index + 2);
        output_col_0 |=
            TBitpacked(output_transform.Run(acc_30, c_out_index + 3))
            << (base_output_index + 3);
        output_col_1 |= TBitpacked(output_transform.Run(acc_01, c_out_index))
                        << base_output_index;
        output_col_1 |=
            TBitpacked(output_transform.Run(acc_11, c_out_index + 1))
            << (base_output_index + 1);
        output_col_1 |=
            TBitpacked(output_transform.Run(acc_21, c_out_index + 2))
            << (base_output_index + 2);
        output_col_1 |=
            TBitpacked(output_transform.Run(acc_31, c_out_index + 3))
            << (base_output_index + 3);

        indirection_ptr -= 2 * filter_size;

        if (group_end_output_channel - c_out_index > 4) {
          c_out_index += 4;
        } else {
          const int gap_to_group_end = group_end_output_channel - c_out_index;
          if (gap_to_group_end < 4) {
            output_col_0 &=
                (TBitpacked(1) << (base_output_index + gap_to_group_end)) - 1;
            output_col_1 &=
                (TBitpacked(1) << (base_output_index + gap_to_group_end)) - 1;
          }
          c_out_index = group_end_output_channel;
          input_depth_offset += input_depth_per_group;
          group_end_output_channel += output_channels_per_group;
        }

        // If on the next iteration we will have 'wrapped around' the output
        // columns, write the bottom halves to the output array.
        if (c_out_index % 16 < base_output_index) {
          *((std::int16_t*)output_ptr_1) = (std::int16_t)output_col_1;
          *((std::int16_t*)output_ptr_0) = (std::int16_t)output_col_0;
          output_col_1 >>= 16;
          output_col_0 >>= 16;
          output_ptr_1 = (TBitpacked*)(((std::int16_t*)output_ptr_1) + 1);
          output_ptr_0 = (TBitpacked*)(((std::int16_t*)output_ptr_0) + 1);
        }
      } while (c_out_index < this->output_channels);

      // If we've got to the end and there are still un-written bits, make sure
      // they get written now.
      if (this->output_channels % 16 > 0) {
        *((std::int16_t*)output_ptr_1) = (std::int16_t)output_col_1;
        *((std::int16_t*)output_ptr_0) = (std::int16_t)output_col_0;
      }
    }
  }
};

}  // namespace indirect_bgemm
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_INDIRECT_BGEMM_KERNEL_4x2_PORTABLE_H_
