#ifndef COMPUTE_ENGINE_CORE_BCONV2D_OPTIMIZED_INDIRECT_BGEMM_H_
#define COMPUTE_ENGINE_CORE_BCONV2D_OPTIMIZED_INDIRECT_BGEMM_H_

#include "larq_compute_engine/core/bconv2d/zero_padding_correction.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace compute_engine {
namespace core {
namespace bconv2d {

template <typename AccumScalar, typename DstScalar>
inline void BConv2DOptimizedIndirectBGEMM(
    const indirect_bgemm::Kernel* kernel, const BConv2DParams* bconv2d_params,
    const RuntimeShape& bitpacked_input_shape, const RuntimeShape& output_shape,
    DstScalar* output_ptr, const float* padding_buffer, const int pad_value) {
  ruy::profiler::ScopeLabel label("BConv2D (optimized, indirect BGEMM)");

  // If writing bitpacked output with a channel count that isn't a multiple of
  // 32 (i.e. where padding bits will be required in the output), fill the
  // output tensor with zeroes in advance so that the BGEMM doesn't have to
  // worry about doing the padding.
  if (std::is_same<DstScalar, TBitpacked>::value &&
      kernel->output_channels % bitpacking_bitwidth != 0) {
    std::fill(
        output_ptr,
        output_ptr + kernel->num_output_pixels *
                         bitpacking::GetBitpackedSize(kernel->output_channels),
        TBitpacked(0));
  }

  kernel->Dispatch(reinterpret_cast<void*>(output_ptr));

  if (std::is_same<DstScalar, float>::value &&
      bconv2d_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      pad_value == 0) {
    ruy::profiler::ScopeLabel label("Zero padding correction");

    const int stride_width = bconv2d_params->stride_width;
    const int stride_height = bconv2d_params->stride_height;
    const int dilation_width_factor = bconv2d_params->dilation_width_factor;
    const int dilation_height_factor = bconv2d_params->dilation_height_factor;
    const int batches = MatchingDim(bitpacked_input_shape, 0, output_shape, 0);
    const int input_depth_per_group =
        bconv2d_params->channels_in / bconv2d_params->groups;
    const int input_width = bitpacked_input_shape.Dims(2);
    const int input_height = bitpacked_input_shape.Dims(1);
    const int filter_height = bconv2d_params->filter_height;
    const int filter_width = bconv2d_params->filter_width;
    const int output_depth = output_shape.Dims(3);
    const int output_width = output_shape.Dims(2);
    const int output_height = output_shape.Dims(1);

    zero_padding_correction::ApplyCorrection(
        batches, input_height, input_width, input_depth_per_group,
        filter_height, filter_width, output_depth, stride_height, stride_width,
        dilation_height_factor, dilation_width_factor,
        reinterpret_cast<float*>(output_ptr), output_height, output_width,
        padding_buffer);
  }
}

}  // namespace bconv2d
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BCONV2D_OPTIMIZED_INDIRECT_BGEMM_H_
