#include <cmath>
#include <cstdint>

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/core/bconv2d_impl_ref.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_impl.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_output_transform_utils.h"
#include "larq_compute_engine/tflite/kernels/bconv2d_params.h"
#include "larq_compute_engine/tflite/kernels/utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace bconv2d {

using ce::core::TBitpacked;

enum class KernelType {
  // kGenericRef: the reference implementation without im2col
  kGenericRef,

  // kRuyOptimized: the Ruy implementation with hand-optimized BGEMM kernels.
  kRuyOptimized,
};

#define LCE_ENSURE_PARAM(conv_params, context, a)                       \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return conv_params;                                               \
    }                                                                   \
  } while (0)

void* Init(TfLiteContext* context, const char* buffer, std::size_t length) {
  auto* conv_params = new TfLiteBConv2DParams{};

  const std::uint8_t* buffer_t = reinterpret_cast<const std::uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  // Read the op's input arguments into the "conv_params" struct

  LCE_ENSURE_PARAM(conv_params, context, !m["stride_height"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["stride_width"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["dilation_height_factor"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["dilation_width_factor"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["padding"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["pad_values"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["channels_in"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context,
                   !m["fused_activation_function"].IsNull());

  conv_params->stride_height = m["stride_height"].AsInt32();
  conv_params->stride_width = m["stride_width"].AsInt32();

  conv_params->dilation_height_factor = m["dilation_height_factor"].AsInt32();
  conv_params->dilation_width_factor = m["dilation_width_factor"].AsInt32();

  conv_params->padding_type = ConvertPadding((Padding)m["padding"].AsInt32());
  conv_params->pad_value = m["pad_values"].AsInt32();
  if (conv_params->pad_value != 0 && conv_params->pad_value != 1) {
    TF_LITE_KERNEL_LOG(context, "Attribute pad_values must be 0 or 1.");
    return conv_params;
  }

  // The input and filters are bitpacked along the channel in axis, which means
  // we cannot infer the 'true' input shape, so there's an explicit integer
  // attribute added to the op in the converter.
  conv_params->channels_in = m["channels_in"].AsInt32();

  conv_params->fused_activation_function = ConvertActivation(
      (ActivationFunctionType)m["fused_activation_function"].AsInt32());
  if (conv_params->padding_type == kTfLitePaddingSame &&
      conv_params->pad_value != 1 &&
      conv_params->fused_activation_function != kTfLiteActNone) {
    TF_LITE_KERNEL_LOG(
        context,
        "Fused activations are only supported with valid or one-padding.");
    return conv_params;
  }

  // It's not possible to return an error code in this method. If we get to here
  // without returning early, initialisation has succeeded without error, and so
  // we set this flag. If it's unset in Prepare, we report the error there.
  conv_params->conv_params_successfully_initialized = true;
  return conv_params;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteBConv2DParams*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  // If an error happened in Init, then return an error code.
  if (!conv_params->conv_params_successfully_initialized) return kTfLiteError;

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  const auto* thresholds = GetInput(context, node, 4);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt32);
  TF_LITE_ENSURE_MSG(context,
                     output->type == kTfLiteInt32 ||
                         output->type == kTfLiteInt8 ||
                         output->type == kTfLiteFloat32,
                     "Supported output types are int8, int32, and float32.");

  // Read the filter dimensions
  conv_params->channels_out = SizeOfDimension(filter, 0);
  conv_params->filter_height = SizeOfDimension(filter, 1);
  conv_params->filter_width = SizeOfDimension(filter, 2);

  // Compute the padding and output values (height, width)
  int out_width, out_height;
  conv_params->padding_values = ComputePaddingHeightWidth(
      conv_params->stride_height, conv_params->stride_width,
      conv_params->dilation_height_factor, conv_params->dilation_width_factor,
      SizeOfDimension(input, 1), SizeOfDimension(input, 2),
      conv_params->filter_height, conv_params->filter_width,
      conv_params->padding_type, &out_height, &out_width);

  if (output->type == kTfLiteInt32) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(thresholds), 1);
    TF_LITE_ENSURE_EQ(context, thresholds->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(thresholds, 0),
                      conv_params->channels_out);
    TF_LITE_ENSURE_MSG(context,
                       conv_params->padding_type != kTfLitePaddingSame ||
                           conv_params->pad_value == 1,
                       "Writing bitpacked output is only supported with "
                       "valid or one-padding.");
  } else {
    TF_LITE_ENSURE_EQ(context, post_activation_multiplier->type,
                      kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, post_activation_bias->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_multiplier), 1);
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_bias), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(post_activation_multiplier, 0),
                      conv_params->channels_out);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(post_activation_bias, 0),
                      conv_params->channels_out);
  }

  if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_MSG(
        context,
        conv_params->padding_type != kTfLitePaddingSame ||
            conv_params->pad_value == 1,
        "8-bit quantization is only supported with valid or one-padding");
  }

  if (kernel_type == KernelType::kGenericRef) {
    TF_LITE_ENSURE_MSG(
        context,
        conv_params->padding_type != kTfLitePaddingSame ||
            conv_params->pad_value == 1,
        "The reference kernel only supports valid or one-padding.");
  }

  // Determine the output dimensions and allocate the output buffer
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(4);
  output_shape->data[0] = SizeOfDimension(input, 0);
  output_shape->data[1] = out_height;
  output_shape->data[2] = out_width;
  output_shape->data[3] =
      output->type == kTfLiteInt32
          ? ce::core::GetBitpackedSize(conv_params->channels_out)
          : conv_params->channels_out;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // Figure out how many temporary buffers we need
  int temporaries_count = 0;

  const bool need_im2col =
      kernel_type == KernelType::kRuyOptimized &&
      (conv_params->stride_width != 1 || conv_params->stride_height != 1 ||
       conv_params->dilation_width_factor != 1 ||
       conv_params->dilation_height_factor != 1 ||
       conv_params->filter_width != 1 || conv_params->filter_height != 1);

  // Pre-allocate temporary tensors
  if (need_im2col) {
    conv_params->im2col_index = temporaries_count++;
  } else {
    conv_params->im2col_index = -1;
  }

  if (temporaries_count != 0) {
    // Allocate int array of that size
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);
  }

  // Now allocate the buffers
  if (need_im2col) {
    // Allocate the im2col tensor if necessary
    if (conv_params->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &conv_params->im2col_id);
      node->temporaries->data[conv_params->im2col_index] =
          conv_params->im2col_id;
    }

    // Resize the im2col tensor
    int channels_in = ce::core::GetBitpackedSize(conv_params->channels_in);
    TfLiteIntArray* im2col_size = TfLiteIntArrayCopy(output_shape);
    im2col_size->data[3] =
        channels_in * conv_params->filter_height * conv_params->filter_width;
    TfLiteTensor* im2col =
        GetTemporary(context, node, conv_params->im2col_index);
    im2col->type = kTfLiteInt32;
    im2col->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, im2col, im2col_size));
  }

  // Prepare could be called multiple times; when the input tensor is resized,
  // we shoud always re-do the one-time setup.
  conv_params->one_time_setup_complete = false;

  return kTfLiteOk;
}

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

inline void GetConvParamsType(const TfLiteBConv2DParams& conv_params,
                              ConvParams& op_params) {
  // Padding
  op_params.padding_type = RuntimePaddingType(conv_params.padding_type);
  op_params.padding_values.width = conv_params.padding_values.width;
  op_params.padding_values.height = conv_params.padding_values.height;

  // Strides
  op_params.stride_height = conv_params.stride_height;
  op_params.stride_width = conv_params.stride_width;

  // Dilations
  op_params.dilation_height_factor = conv_params.dilation_height_factor;
  op_params.dilation_width_factor = conv_params.dilation_width_factor;

  // Activation function
  op_params.quantized_activation_min = conv_params.output_transform_clamp_min;
  op_params.quantized_activation_max = conv_params.output_transform_clamp_max;
}

void OneTimeSetup(TfLiteContext* context, TfLiteNode* node,
                  TfLiteBConv2DParams* params) {
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  const auto* output = GetOutput(context, node, 0);

  // For 'same-zero' padding, compute the padding-correction.
  if (params->padding_type == kTfLitePaddingSame && params->pad_value == 0) {
    params->padding_buffer.resize(ce::core::PaddingFunctor::get_cache_size(
        params->filter_height, params->filter_width, params->channels_out,
        params->dilation_height_factor, params->dilation_width_factor));
    ce::core::PaddingFunctor::cache_correction_values(
        GetTensorData<TBitpacked>(filter), params->filter_height,
        params->filter_width, params->channels_out, params->channels_in,
        params->dilation_height_factor, params->dilation_width_factor,
        GetTensorData<float>(post_activation_multiplier),
        params->padding_buffer.data());
  }

  // If applicable, fuse the back-transformation and int8 scale/zero-point into
  // the output transform multiplier/bias.
  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt8) {
    params->output_transform_multiplier.resize(params->channels_out +
                                               LCE_EXTRA_BYTES / sizeof(float));
    params->output_transform_bias.resize(params->channels_out +
                                         LCE_EXTRA_BYTES / sizeof(float));

    const auto filter_shape = GetTensorShape(GetInput(context, node, 1));
    const std::int32_t backtransform_add =
        filter_shape.Dims(1) * filter_shape.Dims(2) * params->channels_in;
    const double output_scale =
        output->type == kTfLiteInt8 ? output->params.scale : 1.0f;
    const double output_zero_point =
        output->type == kTfLiteInt8 ? output->params.zero_point : 0.0f;

    for (int i = 0; i < params->channels_out; ++i) {
      const double post_mul =
          GetTensorData<float>(post_activation_multiplier)[i];
      const double post_bias = GetTensorData<float>(post_activation_bias)[i];
      params->output_transform_multiplier.at(i) = -1 * post_mul / output_scale;
      params->output_transform_bias.at(i) =
          (post_bias + static_cast<double>(backtransform_add) * post_mul) /
              output_scale +
          output_zero_point;
    }

    std::int32_t nominal_clamp_min, nominal_clamp_max;
    CalculateActivationRange(params->fused_activation_function,
                             &nominal_clamp_min, &nominal_clamp_max);
    nominal_clamp_min = std::max(nominal_clamp_min, -1 * backtransform_add);
    nominal_clamp_max = std::min(nominal_clamp_max, backtransform_add);
    params->output_transform_clamp_min =
        -1 * nominal_clamp_max + backtransform_add;
    params->output_transform_clamp_max =
        -1 * nominal_clamp_min + backtransform_add;
  }

  params->one_time_setup_complete = true;
}

template <typename AccumScalar, typename DstScalar>
void EvalOpt(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  if (!params->one_time_setup_complete) {
    OneTimeSetup(context, node, params);
  }

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TfLiteTensor* im2col = params->im2col_index >= 0
                             ? GetTemporary(context, node, params->im2col_index)
                             : nullptr;

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);
  op_params.input_offset = input->params.zero_point;

  OutputTransform<DstScalar> output_transform;
  GetOutputTransform(output_transform, context, node, params);

  // `BConv2D` wants the *unpacked* output shape.
  auto unpacked_output_shape = GetTensorShape(output);
  unpacked_output_shape.SetDim(3, params->channels_out);

  // We pass the shape of the original unpacked filter, so that all the shape
  // information is correct (number of channels etc), but we pass the packed
  // weights data.
  //     Likewise, we pass the original output shape even if we are going to
  // write bitpacked output directly.
  BConv2D<AccumScalar, DstScalar>(
      op_params, GetTensorShape(input), GetTensorData<TBitpacked>(input),
      GetTensorShape(filter), GetTensorData<TBitpacked>(filter),
      output_transform, unpacked_output_shape, GetTensorData<DstScalar>(output),
      GetTensorShape(im2col), GetTensorData<TBitpacked>(im2col),
      params->padding_buffer.data(), params->pad_value,
      CpuBackendContext::GetFromContext(context));
}

template <typename DstScalar>
void EvalRef(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* packed_filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  if (!params->one_time_setup_complete) {
    OneTimeSetup(context, node, params);
  }

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);

  OutputTransform<DstScalar> output_transform;
  GetOutputTransform(output_transform, context, node, params);

  TfLiteTensor* im2col = nullptr;
  ce::ref::BConv2D<std::int32_t, DstScalar>(
      op_params, GetTensorShape(input), GetTensorData<TBitpacked>(input),
      GetTensorShape(packed_filter), GetTensorData<TBitpacked>(packed_filter),
      output_transform, GetTensorShape(output),
      GetTensorData<DstScalar>(output), GetTensorShape(im2col),
      GetTensorData<TBitpacked>(im2col), nullptr /*padding buffer*/,
      params->pad_value, nullptr /*cpu backend context*/);
}

template <KernelType kernel_type, typename DstScalar>
inline TfLiteStatus EvalChooseKernelType(TfLiteContext* context,
                                         TfLiteNode* node,
                                         TfLiteBConv2DParams* params) {
  if (kernel_type == KernelType::kRuyOptimized) {
#if RUY_PLATFORM_ARM_64
    // On 64 bit Arm only there is an optimised kernel with 16-bit accumulators.
    // It is safe to use this without risk of overflow as long as the maximum
    // value of the convolution (filter height * filter width * input channels,
    // plus some overhead to account for potential padding) is less than 2^16.
    // We will almost always take this path: for a 3x3 filter there would need
    // to be > 7000 input channels to present an overflow risk.
    const int depth =
        params->filter_height * params->filter_width * params->channels_in;
    if (depth + 512 < 1 << 16) {
      EvalOpt<std::int16_t, DstScalar>(context, node, params);
      return kTfLiteOk;
    }
#endif
    // In all other cases, use 32-bit accumulators.
    EvalOpt<std::int32_t, DstScalar>(context, node, params);
    return kTfLiteOk;
  } else if (kernel_type == KernelType::kGenericRef) {
    EvalRef<DstScalar>(context, node, params);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteType output_type = GetOutput(context, node, 0)->type;
  auto* params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);
  if (output_type == kTfLiteFloat32) {
    return EvalChooseKernelType<kernel_type, float>(context, node, params);
  } else if (output_type == kTfLiteInt8) {
    return EvalChooseKernelType<kernel_type, std::int8_t>(context, node,
                                                          params);
  } else if (output_type == kTfLiteInt32) {
    return EvalChooseKernelType<kernel_type, TBitpacked>(context, node, params);
  }
  return kTfLiteError;
}

}  // namespace bconv2d

TfLiteRegistration* Register_BCONV_2D_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericRef>,
      bconv2d::Eval<bconv2d::KernelType::kGenericRef>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D_OPT() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kRuyOptimized>,
      bconv2d::Eval<bconv2d::KernelType::kRuyOptimized>};
  return &r;
}

// Use this registration wrapper to decide which implementation to use.
TfLiteRegistration* Register_BCONV_2D() {
#if defined TFLITE_WITH_RUY
  return Register_BCONV_2D_OPT();
#else
  return Register_BCONV_2D_REF();
#endif
}

}  // namespace tflite
}  // namespace compute_engine
