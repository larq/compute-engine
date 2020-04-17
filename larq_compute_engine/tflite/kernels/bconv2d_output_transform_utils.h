#ifndef LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP
#define LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP

#include "bconv2d_params.h"
#include "larq_compute_engine/core/bconv2d_output_transform.h"
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
void GetBaseParams(TfLiteContext* context, TfLiteNode* node,
                   TfLiteBConv2DParams* params,
                   OutputTransformBase<std::int32_t>& output_transform) {
  auto input_shape = GetTensorShape(GetInput(context, node, 0));
  auto filter_shape = GetTensorShape(GetInput(context, node, 1));
  output_transform.backtransform_add =
      filter_shape.Dims(1) * filter_shape.Dims(2) * params->channels_in;
  output_transform.clamp_min = params->output_activation_min;
  output_transform.clamp_max = params->output_activation_max;
}

// Fill the OutputTransform values for float outputs
void GetOutputTransform(
    TfLiteContext* context, TfLiteNode* node, TfLiteBConv2DParams* params,
    OutputTransform<std::int32_t, float>& output_transform) {
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

// Fill the OutputTransform values for bitpacked int32 outputs
void GetOutputTransform(
    TfLiteContext* context, TfLiteNode* node, TfLiteBConv2DParams* params,
    OutputTransform<std::int32_t, std::int32_t>& output_transform) {
  GetBaseParams(context, node, params, output_transform);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  if (post_activation_multiplier->type == kTfLiteFloat32 &&
      post_activation_bias->type == kTfLiteFloat32) {
    output_transform.post_activation_multiplier =
        GetTensorData<float>(post_activation_multiplier);
    output_transform.post_activation_bias =
        GetTensorData<float>(post_activation_bias);
  } else {
    // When the post data was stored in int8, then SetupQuantization will have
    // converted it to float
    output_transform.post_activation_multiplier =
        params->scaled_post_activation_multiplier.data();
    output_transform.post_activation_bias =
        params->scaled_post_activation_bias.data();
  }
}

// Fill the OutputTransform values for int8 outputs
void GetOutputTransform(
    TfLiteContext* context, TfLiteNode* node, TfLiteBConv2DParams* params,
    OutputTransform<std::int32_t, std::int8_t>& output_transform) {
  GetBaseParams(context, node, params, output_transform);
#ifdef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
  output_transform.effective_post_activation_multiplier =
      params->scaled_post_activation_multiplier.data();
  output_transform.effective_post_activation_bias =
      params->scaled_post_activation_bias.data();
  output_transform.output_zero_point =
      GetOutput(context, node, 0)->params.zero_point;
#else
  output_transform.output_multiplier = params->output_multiplier.data();
  output_transform.output_shift = params->output_shift.data();
  output_transform.output_effective_zero_point =
      params->output_zero_point.data();
#endif
}

}  // namespace bconv2d
}  // namespace tflite
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_KERNELS_BCONV2D_OUTPUT_TRANSFORM_SETUP
