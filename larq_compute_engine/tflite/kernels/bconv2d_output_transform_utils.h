#ifndef LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP
#define LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP

#include "larq_compute_engine/core/bconv2d_output_transform.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_params.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {
namespace bconv2d {

using compute_engine::core::OutputTransform;

// Fill in the OutputTransform values for float and/or int8 outputs
template <typename DstScalar>
void GetOutputTransform(OutputTransform<DstScalar>& output_transform,
                        TfLiteContext* context, TfLiteNode* node,
                        TfLiteBConv2DParams* params) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");
  output_transform.clamp_min = params->output_transform_clamp_min;
  output_transform.clamp_max = params->output_transform_clamp_max;
  output_transform.multiplier = params->output_transform_multiplier.data();
  output_transform.bias = params->output_transform_bias.data();
}

// Fill in the OutputTransform values for bitpacked outputs
void GetOutputTransform(OutputTransform<TBitpacked>& output_transform,
                        TfLiteContext* context, TfLiteNode* node,
                        TfLiteBConv2DParams* params) {
  const auto* thresholds = GetInput(context, node, 4);
  output_transform.thresholds = GetTensorData<std::int32_t>(thresholds);
}

}  // namespace bconv2d
}  // namespace tflite
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP
