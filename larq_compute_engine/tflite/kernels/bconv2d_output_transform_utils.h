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
using compute_engine::core::OutputTransformBase;

// Fill the parts of the OutputTransform struct that are common to each
// destination type
template <typename AccumScalar>
void GetBaseParams(TfLiteContext* context, TfLiteNode* node,
                   TfLiteBConv2DParams* params,
                   OutputTransformBase<AccumScalar>& output_transform) {
  static_assert(std::is_same<AccumScalar, std::int32_t>::value ||
                    std::is_same<AccumScalar, std::int16_t>::value,
                "AccumScalar must be int32 or int16");
  auto filter_shape = GetTensorShape(GetInput(context, node, 1));
  output_transform.backtransform_add =
      filter_shape.Dims(1) * filter_shape.Dims(2) * params->channels_in;
  output_transform.clamp_min = params->output_activation_min;
  output_transform.clamp_max = params->output_activation_max;
}

// Fill the OutputTransform values for float outputs
template <typename AccumScalar>
void GetOutputTransform(TfLiteContext* context, TfLiteNode* node,
                        TfLiteBConv2DParams* params,
                        OutputTransform<AccumScalar, float>& output_transform) {
  GetBaseParams(context, node, params, output_transform);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  TF_LITE_ASSERT_EQ(post_activation_multiplier->type, kTfLiteFloat32);
  TF_LITE_ASSERT_EQ(post_activation_bias->type, kTfLiteFloat32);
  output_transform.post_activation_multiplier =
      GetTensorData<float>(post_activation_multiplier);
  output_transform.post_activation_bias =
      GetTensorData<float>(post_activation_bias);
}

// Fill the OutputTransform values for bitpacked outputs
template <typename AccumScalar>
void GetOutputTransform(
    TfLiteContext* context, TfLiteNode* node, TfLiteBConv2DParams* params,
    OutputTransform<AccumScalar, TBitpacked>& output_transform) {
  const auto* thresholds = GetInput(context, node, 4);
  output_transform.thresholds = GetTensorData<AccumScalar>(thresholds);
}

// Fill the OutputTransform values for int8 outputs
template <typename AccumScalar>
void GetOutputTransform(
    TfLiteContext* context, TfLiteNode* node, TfLiteBConv2DParams* params,
    OutputTransform<AccumScalar, std::int8_t>& output_transform) {
  GetBaseParams(context, node, params, output_transform);
  output_transform.effective_post_activation_multiplier =
      params->scaled_post_activation_multiplier.data();
  output_transform.effective_post_activation_bias =
      params->scaled_post_activation_bias.data();
}

}  // namespace bconv2d
}  // namespace tflite
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP
