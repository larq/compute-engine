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
    const indirect_bgemm::IndirectBGEMMKernel<DstScalar> kernel,
    const BConv2DParams* bconv2d_params,
    const RuntimeShape& bitpacked_input_shape, const RuntimeShape& output_shape,
    const OutputTransform<DstScalar>& output_transform,
    const TBitpacked* packed_weights, const TBitpacked** indirection_buffer,
    DstScalar* output_data, const float* padding_buffer, const int pad_value) {
  TF_LITE_ASSERT_EQ(bitpacked_input_shape.DimensionsCount(), 4);
  TF_LITE_ASSERT_EQ(output_shape.DimensionsCount(), 4);

  ruy::profiler::ScopeLabel label("BConv2D (optimized, indirect BGEMM)");

  const std::int32_t conv_kernel_size =
      bconv2d_params->filter_height * bconv2d_params->filter_width;
  const std::int32_t bitpacked_input_channels = bitpacked_input_shape.Dims(3);
  const std::int32_t output_size =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  const std::int32_t output_channels = bconv2d_params->channels_out;

  indirect_bgemm::RunKernel(kernel, conv_kernel_size, bitpacked_input_channels,
                            output_size, output_channels, output_transform,
                            packed_weights, indirection_buffer, output_data);

  if (std::is_same<DstScalar, float>::value &&
      bconv2d_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      pad_value == 0) {
    ruy::profiler::ScopeLabel label("Zero padding correction");

    const int stride_width = bconv2d_params->stride_width;
    const int stride_height = bconv2d_params->stride_height;
    const int dilation_width_factor = bconv2d_params->dilation_width_factor;
    const int dilation_height_factor = bconv2d_params->dilation_height_factor;
    const int batches = MatchingDim(bitpacked_input_shape, 0, output_shape, 0);
    const int input_depth = bconv2d_params->channels_in;
    const int input_width = bitpacked_input_shape.Dims(2);
    const int input_height = bitpacked_input_shape.Dims(1);
    const int filter_height = bconv2d_params->filter_height;
    const int filter_width = bconv2d_params->filter_width;
    const int output_depth = output_shape.Dims(3);
    const int output_width = output_shape.Dims(2);
    const int output_height = output_shape.Dims(1);

    zero_padding_correction::ApplyCorrection(
        batches, input_height, input_width, input_depth, filter_height,
        filter_width, output_depth, stride_height, stride_width,
        dilation_height_factor, dilation_width_factor,
        reinterpret_cast<float*>(output_data), output_height, output_width,
        padding_buffer);
  }
}

}  // namespace bconv2d
}  // namespace core
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_CORE_BCONV2D_OPTIMIZED_INDIRECT_BGEMM_H_
