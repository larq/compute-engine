#include <cmath>
#include <cstdint>

#include "bconv2d_impl.h"
#include "bconv2d_output_transform_utils.h"
#include "bconv2d_params.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/bconv2d_impl_ref.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace bconv2d {

using ::compute_engine::core::Layout;

enum class KernelType {
  // kGenericRef: the impl. path using reference implementation without im2col
  kGenericRef,

  // kGenericOptimized: the impl. path using reference BGEMM kernels in RUY.
  // TODO(arashb): the generic optimized needs to be redirected to the
  // reference impl. of RUY kernels.
  kGenericOptimized,

  // kRuyOptimized: the impl. path using RUY framework with hand-optimized
  // BGEMM kernels.
  kRuyOptimized,
};

inline void decide_bitpack_before_im2col(KernelType kernel_type,
                                         TfLiteBConv2DParams* conv_params) {
  if (kernel_type == KernelType::kGenericRef ||
      conv_params->read_bitpacked_input ||
      conv_params->channels_in >= conv_params->bitpacking_bitwidth / 4) {
    conv_params->bitpack_before_im2col = true;
  } else {
    conv_params->bitpack_before_im2col = false;
  }
}

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

  // reading the op's input arguments into the "conv_params" struct
  LCE_ENSURE_PARAM(conv_params, context, !m["stride_height"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["stride_width"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["dilation_height_factor"].IsNull());
  LCE_ENSURE_PARAM(conv_params, context, !m["dilation_width_factor"].IsNull());

  // reading strides
  conv_params->stride_height = m["stride_height"].AsInt32();
  conv_params->stride_width = m["stride_width"].AsInt32();
  // reading dilations
  conv_params->dilation_height_factor = m["dilation_height_factor"].AsInt32();
  conv_params->dilation_width_factor = m["dilation_width_factor"].AsInt32();

  // reading padding
  if (m["padding"].ToString() == "VALID" ||
      m["padding"].ToString() == "valid") {
    conv_params->padding_type = kTfLitePaddingValid;
  } else if (m["padding"].ToString() == "SAME" ||
             m["padding"].ToString() == "same") {
    conv_params->padding_type = kTfLitePaddingSame;
  } else {
    context->ReportError(context, "Invalid padding attribute.");
    return conv_params;
  }

  // Read fused activation
  if (m["fused_activation_function"].IsNull() ||
      m["fused_activation_function"].ToString() == "" ||
      m["fused_activation_function"].ToString() == "NONE") {
    conv_params->fused_activation_function = kTfLiteActNone;
  } else if (m["fused_activation_function"].ToString() == "RELU") {
    conv_params->fused_activation_function = kTfLiteActRelu;
  } else if (m["fused_activation_function"].ToString() == "RELU_N1_TO_1") {
    conv_params->fused_activation_function = kTfLiteActRelu1;
  } else if (m["fused_activation_function"].ToString() == "RELU6") {
    conv_params->fused_activation_function = kTfLiteActRelu6;
  } else {
    context->ReportError(context,
                         "Invalid value for fused_activation_function.");
    return conv_params;
  }
  conv_params->pad_value =
      m["pad_values"].IsNull() ? 0 : m["pad_values"].AsInt32();
  if (conv_params->pad_value != 0 && conv_params->pad_value != 1) {
    context->ReportError(context, "Attribute pad_values must be 0 or 1.");
    return conv_params;
  }

  // If we are reading bitpacked input then both the input tensor and the
  // filters are bitpacked along the (input) channels axis. This means that we
  // cannot infer the 'true' input shape, and so we have to add an explicit
  // integer attribute to the op in the converter.
  if (!m["channels_in"].IsNull()) {
    conv_params->channels_in = m["channels_in"].AsInt32();
  }

  if (conv_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      conv_params->pad_value != 1 &&
      conv_params->fused_activation_function != kTfLiteActNone) {
    context->ReportError(
        context,
        "Fused activations are only supported with valid or one-padding.");
    return conv_params;
  }

  // We cannot return an error code here, so we set this flag and return an
  // error code in Prepare
  conv_params->conv_params_initialized = true;
  return conv_params;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteBConv2DParams*>(buffer);
}

TfLiteStatus Prepare(KernelType kernel_type,
                     const int default_bitpacking_bitwidth,
                     TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  // If an error happened in Init, then report an error message
  if (!conv_params->conv_params_initialized) return kTfLiteError;

  TF_LITE_ENSURE(context, node->inputs->size == 5);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  const auto* thresholds = GetInput(context, node, 4);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);

  conv_params->read_bitpacked_input = input->type == kTfLiteInt32;
  conv_params->write_bitpacked_output = output->type == kTfLiteInt32;

  if (conv_params->write_bitpacked_output) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(thresholds), 1);
  } else {
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_multiplier), 1);
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_bias), 1);
  }

  if (conv_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      conv_params->pad_value != 1 && conv_params->write_bitpacked_output) {
    context->ReportError(context,
                         "Writing bitpacked output is only supported with "
                         "valid or one-padding.");
    return kTfLiteError;
  }

  // Read the input dimensions. TF Lite has the same input format as TensorFlow:
  // (B, H, W, Ci).
  conv_params->batch = input->dims->data[0];
  conv_params->input_height = input->dims->data[1];
  conv_params->input_width = input->dims->data[2];
  if (!conv_params->read_bitpacked_input) {
    TF_LITE_ENSURE_EQ(context, conv_params->channels_in, input->dims->data[3]);
  } else if (conv_params->channels_in == 0) {
    // We don't expect this branch to ever be taken because the `channels_in`
    // attribute was added to the converter at the same time that support for
    // bitpacked activations was added, but just in case we don't have a value
    // we should throw here.
    context->ReportError(context,
                         "Cannot read bitpacked input unless the `channels_in` "
                         "attribute is set in the converter.");
    return kTfLiteError;
  }

  if (conv_params->write_bitpacked_output) {
    TF_LITE_ENSURE_EQ(context, thresholds->type, kTfLiteInt32);
  } else {
    // For 8-bit quantized networks, we support both int8 and float32
    // post_activation_ values
    TF_LITE_ENSURE(context,
                   post_activation_multiplier->type == kTfLiteFloat32 ||
                       post_activation_multiplier->type == kTfLiteInt8);
    TF_LITE_ENSURE(context, post_activation_bias->type == kTfLiteFloat32 ||
                                post_activation_bias->type == kTfLiteInt8);

    // In case that our network is not 8-bit quantized, we do require float
    // values
    if (input->type == kTfLiteFloat32 || output->type == kTfLiteFloat32) {
      TF_LITE_ENSURE_EQ(context, post_activation_multiplier->type,
                        kTfLiteFloat32);
      TF_LITE_ENSURE_EQ(context, post_activation_bias->type, kTfLiteFloat32);
    }
  }

  if (conv_params->read_bitpacked_input) {
    // The kernels support only 32-bit bitpacking when reading bitpacked input.
    conv_params->bitpacking_bitwidth = 32;
  } else {
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 || input->type == kTfLiteFloat32);

    if (input->type == kTfLiteInt8 &&
        conv_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
        conv_params->pad_value != 1) {
      context->ReportError(
          context,
          "8-bit quantization is only supported with valid or one-padding");
      return kTfLiteError;
    }

    // TODO: more intelligent selection of the parameters `bitpacking_bitwidth`
    //       and `bitpack_before_im2col` based on benchmarking results
    //       (as in https://github.com/larq/compute-engine/issues/290).
    conv_params->bitpacking_bitwidth = default_bitpacking_bitwidth;
    conv_params->channels_in = input->dims->data[3];
  }

  if (!conv_params->write_bitpacked_output) {
    if (conv_params->read_bitpacked_input) {
      TF_LITE_ENSURE(context, output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteFloat32);

    } else {
      // We're not reading or writing bitpacked output, so if the input is
      // float, so should the output, and likewise for int8.
      TF_LITE_ENSURE_EQ(context, output->type, input->type);
    }
  }

  // reading the filter dimensions
  conv_params->channels_out = filter->dims->data[0];
  conv_params->filter_height = filter->dims->data[1];
  conv_params->filter_width = filter->dims->data[2];

  if (filter->type == kTfLiteFloat32) {
    conv_params->filter_format = ce::core::FilterFormat::OHWI;
    TF_LITE_ENSURE_EQ(context, conv_params->channels_in, filter->dims->data[3]);
  } else if (filter->type == kTfLiteInt32) {
    conv_params->filter_format = ce::core::FilterFormat::OHWI_PACKED;
  } else {
    context->ReportError(context, "Invalid filter format.");
    return kTfLiteError;
  }

  if (conv_params->write_bitpacked_output) {
    TF_LITE_ENSURE_EQ(context, thresholds->dims->data[0],
                      conv_params->channels_out);
  } else {
    TF_LITE_ENSURE_EQ(context, post_activation_multiplier->dims->data[0],
                      conv_params->channels_out);
    TF_LITE_ENSURE_EQ(context, post_activation_bias->dims->data[0],
                      conv_params->channels_out);
  }

  // computing the padding and output values (height, width)
  int out_width, out_height;
  conv_params->padding_values = ComputePaddingHeightWidth(
      conv_params->stride_height, conv_params->stride_width,
      conv_params->dilation_height_factor, conv_params->dilation_width_factor,
      conv_params->input_height, conv_params->input_width,
      conv_params->filter_height, conv_params->filter_width,
      conv_params->padding_type, &out_height, &out_width);

  conv_params->out_width = out_width;
  conv_params->out_height = out_height;

  CalculateActivationRange(conv_params->fused_activation_function,
                           &conv_params->output_activation_min,
                           &conv_params->output_activation_max);

  if (input->type == kTfLiteInt8) {
    // 8-bit quantized input and output
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
  }
  if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);

    // We resize the arrays here
    // They will be filled in OneTimeSetup
#ifndef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
    conv_params->output_multiplier.resize(conv_params->channels_out);
    conv_params->output_shift.resize(conv_params->channels_out);
    conv_params->output_zero_point.resize(conv_params->channels_out);
#endif
  }

  if (!conv_params->write_bitpacked_output) {
    conv_params->scaled_post_activation_multiplier.resize(
        conv_params->channels_out);
    conv_params->scaled_post_activation_bias.resize(conv_params->channels_out);
  }

  // determine the output dimensions
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(4);
  output_shape->data[0] = conv_params->batch;
  output_shape->data[1] = conv_params->out_height;
  output_shape->data[2] = conv_params->out_width;
  if (conv_params->write_bitpacked_output) {
    // If we write bitpacked output, we use 32-bit bitpacking
    output_shape->data[3] = (conv_params->channels_out + 31) / 32;
  } else {
    output_shape->data[3] = conv_params->channels_out;
  }

  // allocate the output buffer
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));

  // Decide if we do bitpacking before or after im2col
  decide_bitpack_before_im2col(kernel_type, conv_params);

  if (kernel_type == KernelType::kGenericRef) {
    // We require 32-bit bitpacking in the reference implementation
    TF_LITE_ENSURE_EQ(context, conv_params->bitpacking_bitwidth, 32);
    // We only support one-padding or valid-padding in the reference
    // implementation.
    TF_LITE_ENSURE(context, !(conv_params->pad_value == 0 &&
                              conv_params->padding_type ==
                                  TfLitePadding::kTfLitePaddingSame));
  }

  // Figure out how many temporary buffers we need
  int temporaries_count = 0;

  // pre-allocate temporary tensors for optimized version
  if (kernel_type == KernelType::kRuyOptimized) {
    conv_params->need_im2col =
        (conv_params->stride_width != 1 || conv_params->stride_height != 1 ||
         conv_params->dilation_width_factor != 1 ||
         conv_params->dilation_height_factor != 1 ||
         conv_params->filter_width != 1 || conv_params->filter_height != 1);

    if (conv_params->need_im2col) {
      conv_params->im2col_index = temporaries_count++;
    }
  }

  // See if we need a temporary buffer for bitpacked activations
  if (!conv_params->read_bitpacked_input) {
    conv_params->packed_input_index = temporaries_count++;
  }

  if (temporaries_count != 0) {
    // Allocate int array of that size
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);
  }

  // Now allocate the buffers
  if (conv_params->need_im2col) {
    if (conv_params->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &conv_params->im2col_id);
      node->temporaries->data[conv_params->im2col_index] =
          conv_params->im2col_id;
    }

    // Resize the im2col tensor
    int channels_in = conv_params->bitpack_before_im2col
                          ? ((conv_params->channels_in +
                              conv_params->bitpacking_bitwidth - 1) /
                             conv_params->bitpacking_bitwidth)
                          : conv_params->channels_in;

    // determine the im2col buffer size
    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);
    im2col_size->data[0] = conv_params->batch;
    im2col_size->data[1] = conv_params->out_height;
    im2col_size->data[2] = conv_params->out_width;
    im2col_size->data[3] =
        channels_in * conv_params->filter_height * conv_params->filter_width;

    // get the pointer to im2col tensor
    TfLiteTensor* im2col =
        GetTemporary(context, node, conv_params->im2col_index);

    // Determine the type
    if (conv_params->bitpack_before_im2col) {
      switch (conv_params->bitpacking_bitwidth) {
        case 32:
          im2col->type = kTfLiteInt32;
          break;
        case 64:
          im2col->type = kTfLiteInt64;
          break;
        default:
          TF_LITE_ENSURE(context, false);
          break;
      }
    } else {
      // im2col before bitpacking so use the same type as the input
      im2col->type = input->type;
    }
    im2col->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, im2col, im2col_size));
  }

  if (!conv_params->read_bitpacked_input) {
    if (conv_params->packed_input_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &conv_params->packed_input_id);
      node->temporaries->data[conv_params->packed_input_index] =
          conv_params->packed_input_id;
    }
    // Determine the size
    int flat_size = 0;
    if (conv_params->bitpack_before_im2col) {
      const int packed_channels =
          ((conv_params->channels_in + conv_params->bitpacking_bitwidth - 1) /
           conv_params->bitpacking_bitwidth);
      flat_size = conv_params->batch * conv_params->input_height *
                  conv_params->input_width * packed_channels;
    } else {
      const int packed_depth =
          (conv_params->channels_in * conv_params->filter_height *
               conv_params->filter_width +
           conv_params->bitpacking_bitwidth - 1) /
          conv_params->bitpacking_bitwidth;

      flat_size = conv_params->batch * conv_params->out_height *
                  conv_params->out_width * packed_depth;
    }
    // We will simply request a flat tensor
    TfLiteIntArray* packed_input_size = TfLiteIntArrayCreate(1);
    packed_input_size->data[0] = flat_size;

    // Get the newly created tensor
    TfLiteTensor* packed_input =
        GetTemporary(context, node, conv_params->packed_input_index);

    // Set its type
    switch (conv_params->bitpacking_bitwidth) {
      case 32:
        packed_input->type = kTfLiteInt32;
        break;
      case 64:
        packed_input->type = kTfLiteInt64;
        break;
      default:
        TF_LITE_ENSURE(context, false);
        break;
    }
    packed_input->allocation_type = kTfLiteArenaRw;
    // Request a resize
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, packed_input,
                                                     packed_input_size));
  }

  // Prepare could be called multiple times, when the input tensor is resized,
  // so we always reset these flags
  conv_params->is_quantization_initialized = false;
  conv_params->is_padding_correction_cached = false;
  conv_params->is_filter_repacked = false;

  return kTfLiteOk;
}

template <KernelType kernel_type, int default_bitpacking_bitwidth>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, default_bitpacking_bitwidth, context, node);
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
  // padding
  op_params.padding_type = RuntimePaddingType(conv_params.padding_type);
  op_params.padding_values.width = conv_params.padding_values.width;
  op_params.padding_values.height = conv_params.padding_values.height;

  // strides
  op_params.stride_height = conv_params.stride_height;
  op_params.stride_width = conv_params.stride_width;

  // dilations
  op_params.dilation_height_factor = conv_params.dilation_height_factor;
  op_params.dilation_width_factor = conv_params.dilation_width_factor;

  // Activation function
  op_params.quantized_activation_min = conv_params.output_activation_min;
  op_params.quantized_activation_max = conv_params.output_activation_max;
}

float GetTensorValue(const TfLiteTensor* tensor, int index) {
  if (tensor->type == kTfLiteFloat32) {
    return GetTensorData<float>(tensor)[index];
  }
  TF_LITE_ASSERT_EQ(tensor->type, kTfLiteInt8);
  return tensor->params.scale *
         (static_cast<int32_t>(GetTensorData<std::int8_t>(tensor)[index]) -
          tensor->params.zero_point);
}

void SetupQuantization(TfLiteContext* context, TfLiteNode* node,
                       TfLiteBConv2DParams* params) {
  const auto* output = GetOutput(context, node, 0);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);

  // Not required when writing bitpacked output
  if (output->type == kTfLiteInt32) return;

  // Step 1
  // If the post_activation_ data is stored in int8, then convert it to float
  if (post_activation_multiplier->type != kTfLiteFloat32 ||
      post_activation_bias->type != kTfLiteFloat32) {
    for (int i = 0; i < params->channels_out; ++i) {
      params->scaled_post_activation_multiplier[i] =
          GetTensorValue(post_activation_multiplier, i);
      params->scaled_post_activation_bias[i] =
          GetTensorValue(post_activation_bias, i);
    }
  }

  if (output->type != kTfLiteInt8) {
    return;
  }

  // Step 2
  // If the output is 8-bit quantized, then rescale these values by the output
  // scale
  const double output_scale = static_cast<double>(output->params.scale);
  for (int i = 0; i < params->channels_out; ++i) {
    const double post_mul = GetTensorValue(post_activation_multiplier, i);
    const double post_bias = GetTensorValue(post_activation_bias, i);

#ifdef LCE_RUN_OUTPUT_TRANSFORM_IN_FLOAT
    params->scaled_post_activation_multiplier[i] = post_mul / output_scale;
    params->scaled_post_activation_bias[i] = post_bias / output_scale;
#else
    const double effective_scale = post_mul / output_scale;

    const std::int32_t effective_zero_point =
        output->params.zero_point + std::lround(post_bias / output_scale);

    std::int32_t significand;
    std::int32_t shift;
    QuantizeMultiplier(effective_scale, &significand, &shift);

    params->output_multiplier[i] = significand;
    params->output_shift[i] = shift;
    params->output_zero_point[i] = effective_zero_point;
#endif
  }
}

// Helper to get the type to unpack to
template <typename SrcScalar>
struct GetUnpackType {};

template <>
struct GetUnpackType<float> {
  using type = float;
};

template <>
struct GetUnpackType<std::int32_t> {
  using type = float;
};

template <>
struct GetUnpackType<std::int8_t> {
  using type = std::int8_t;
};

template <class SrcScalar, class TBitpacked>
void OneTimeSetup(TfLiteContext* context, TfLiteNode* node,
                  TfLiteBConv2DParams* params) {
  if (!params->is_filter_repacked || !params->is_padding_correction_cached) {
    const auto* filter = GetInput(context, node, 1);
    const auto* post_activation_multiplier = GetInput(context, node, 2);

    const std::uint32_t* filter_flatbuffer =
        GetTensorData<std::uint32_t>(filter);

    using UnpackType = typename GetUnpackType<SrcScalar>::type;

    const UnpackType* filter_unpacked = nullptr;

    if (params->filter_format == ce::core::FilterFormat::OHWI_PACKED) {
      // First unpack the filter to SrcScalar
      int cols = params->channels_in;
      int rows =
          params->channels_out * params->filter_height * params->filter_width;

      // This vector is declared static, so that it will be shared by all nodes.
      static std::vector<UnpackType> unpacked_weights;
      unpacked_weights.resize(rows * cols);

      ce::core::unpack_matrix(filter_flatbuffer, rows, cols,
                              unpacked_weights.data());

      filter_unpacked = unpacked_weights.data();
    } else {
      // Filter was already unpacked
      filter_unpacked = GetTensorData<UnpackType>(filter);
    }

    // Fill the zero-padding cache
    if (!params->is_padding_correction_cached &&
        (params->padding_type == TfLitePadding::kTfLitePaddingSame &&
         params->pad_value == 0)) {
      using PaddingFunctor =
          ce::core::PaddingFunctor<float, UnpackType, float, float,
                                   ce::core::FilterFormat::OHWI>;
      PaddingFunctor padding_functor;

      std::size_t padding_cache_size = padding_functor.get_cache_size(
          params->filter_height, params->filter_width, params->channels_out,
          params->dilation_height_factor, params->dilation_width_factor);

      params->padding_buffer.resize(padding_cache_size);

      padding_functor.cache_correction_values(
          filter_unpacked, params->filter_height, params->filter_width,
          params->channels_out, params->channels_in,
          params->dilation_height_factor, params->dilation_width_factor,
          GetTensorData<float>(post_activation_multiplier),
          params->padding_buffer.data());
    }
    params->is_padding_correction_cached = true;

    // Repack the filters. They have shape
    // [output channels, height, width, input channels]
    // and we now view it as a matrix of shape
    // bitpack first: [output channels * height * width, input_channels]
    // im2col first:  [output channels, height * width * input_channels]
    // and bitpack it along the last dimension

    int cols, rows;
    if (params->bitpack_before_im2col) {
      cols = params->channels_in;
      rows =
          params->channels_out * params->filter_height * params->filter_width;
    } else {
      cols = params->channels_in * params->filter_height * params->filter_width;
      rows = params->channels_out;
    }

    std::vector<TBitpacked> filter_data_bp(
        ce::core::GetPackedMatrixSize<TBitpacked>(rows, cols));
    if (std::is_same<SrcScalar, std::int32_t>::value) {
      // If the input is already bitpacked, we require canonical order
      ce::core::packbits_matrix<ce::core::BitpackOrder::Canonical>(
          filter_unpacked, rows, cols, filter_data_bp.data());
    } else {
      // Input is int8 or float, use the optimized order
      ce::core::packbits_matrix<ce::core::BitpackOrder::Optimized>(
          filter_unpacked, rows, cols, filter_data_bp.data());
    }

    std::size_t num_bytes = filter_data_bp.size() * sizeof(TBitpacked);

    params->filter_packed.resize(num_bytes);
    memcpy(params->filter_packed.data(), filter_data_bp.data(), num_bytes);

    params->is_filter_repacked = true;
  }
  if (!params->is_quantization_initialized) {
    SetupQuantization(context, node, params);
    params->is_quantization_initialized = true;
  }
}

template <typename SrcScalar, typename TBitpacked, typename AccumScalar,
          typename DstScalar>
void EvalOpt(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  OneTimeSetup<SrcScalar, TBitpacked>(context, node, params);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TfLiteTensor* im2col = params->need_im2col
                             ? GetTemporary(context, node, params->im2col_index)
                             : nullptr;

  TBitpacked* packed_input_data =
      !params->read_bitpacked_input
          ? GetTensorData<TBitpacked>(
                GetTemporary(context, node, params->packed_input_index))
          : nullptr;

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);
  op_params.input_offset = input->params.zero_point;

  OutputTransform<AccumScalar, DstScalar> output_transform;
  GetOutputTransform(context, node, params, output_transform);

  // `BConv2D` wants the *unpacked* filter and output shape.
  auto unpacked_filter_shape = GetTensorShape(filter);
  unpacked_filter_shape.SetDim(3, params->channels_in);
  auto unpacked_output_shape = GetTensorShape(output);
  unpacked_output_shape.SetDim(3, params->channels_out);

  // We pass the shape of the original unpacked filter, so that all the shape
  // information is correct (number of channels etc), but we pass the packed
  // weights data.
  //     Likewise, we pass the original output shape even if we are going to
  // write bitpacked output directly.
  BConv2D<SrcScalar, TBitpacked, AccumScalar, DstScalar>(
      op_params, GetTensorShape(input), GetTensorData<SrcScalar>(input),
      packed_input_data, unpacked_filter_shape,
      reinterpret_cast<TBitpacked*>(params->filter_packed.data()),
      output_transform, unpacked_output_shape, GetTensorData<DstScalar>(output),
      GetTensorShape(im2col), GetTensorData<SrcScalar>(im2col),
      params->bitpack_before_im2col, params->padding_buffer.data(),
      params->pad_value, params->read_bitpacked_input,
      CpuBackendContext::GetFromContext(context));
}

template <typename SrcScalar, typename TBitpacked, typename DstScalar>
void EvalRef(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* packed_filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  if (!params->is_quantization_initialized) {
    SetupQuantization(context, node, params);
    params->is_quantization_initialized = true;
  }

  auto input_shape = GetTensorShape(input);
  const auto packed_filter_shape = GetTensorShape(packed_filter);

  // Bitpack the input data, unless we're reading bitpacked input.
  auto input_data = GetTensorData<SrcScalar>(input);
  const TBitpacked* packed_input_data;
  RuntimeShape packed_input_shape;
  if (params->read_bitpacked_input) {
    packed_input_shape.ReplaceWith(4, input_shape.DimsData());
    packed_input_data = reinterpret_cast<const TBitpacked*>(input_data);
  } else {
    TfLiteTensor* packed_input =
        GetTemporary(context, node, params->packed_input_index);
    ruy::profiler::ScopeLabel label("Bitpack activations (EvalRef)");
    ce::core::packbits_tensor<ce::core::BitpackOrder::Canonical>(
        input_shape, input_data, input->params.zero_point, packed_input_shape,
        GetTensorData<TBitpacked>(packed_input));
    packed_input_data = GetTensorData<TBitpacked>(packed_input);
  }

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);

  OutputTransform<std::int32_t, DstScalar> output_transform;
  GetOutputTransform(context, node, params, output_transform);

  TfLiteTensor* im2col = nullptr;
  ce::ref::BConv2D<TBitpacked, std::int32_t, DstScalar>(
      op_params, packed_input_shape, packed_input_data, packed_filter_shape,
      GetTensorData<TBitpacked>(packed_filter), output_transform,
      GetTensorShape(output), GetTensorData<DstScalar>(output),
      GetTensorShape(im2col), GetTensorData<TBitpacked>(im2col),
      false /*bitpack before im2col*/, nullptr /*padding buffer*/,
      params->pad_value, nullptr /*cpu backend context*/);
}

template <KernelType kernel_type, typename SrcScalar, typename DstScalar,
          typename TBitpacked>
TfLiteStatus EvalChooseKernelType(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteBConv2DParams* params) {
  if (kernel_type == KernelType::kRuyOptimized) {
#if RUY_PLATFORM_ARM_64
    // On 64 bit Arm only there is an optimised kernel for 16-bit accumulators
    // and float output. It is safe to use this without risk of overflow as long
    // as the maximum value of the convolution (filter height * filter width *
    // input channels, plus some overhead to account for potential padding) is
    // less than 2^16. We will almost always take this path: for a 3x3 filter
    // there would need to be > 7000 input channels to present an overflow risk.
    const int depth =
        params->filter_height * params->filter_width * params->channels_in;
    if (std::is_same<DstScalar, float>::value && depth + 512 < 1 << 16) {
      EvalOpt<SrcScalar, TBitpacked, std::int16_t, DstScalar>(context, node,
                                                              params);
      return kTfLiteOk;
    }
#endif
    // In all other cases, use 32-bit accumulators.
    EvalOpt<SrcScalar, TBitpacked, std::int32_t, DstScalar>(context, node,
                                                            params);
    return kTfLiteOk;
  } else if (kernel_type == KernelType::kGenericRef) {
    EvalRef<SrcScalar, TBitpacked, DstScalar>(context, node, params);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

template <KernelType kernel_type, typename SrcScalar, typename DstScalar>
TfLiteStatus EvalChooseBitpackType(TfLiteContext* context, TfLiteNode* node,
                                   TfLiteBConv2DParams* params) {
  switch (params->bitpacking_bitwidth) {
    case 32:
      return EvalChooseKernelType<kernel_type, SrcScalar, DstScalar,
                                  std::uint32_t>(context, node, params);
    case 64:
      return EvalChooseKernelType<kernel_type, SrcScalar, DstScalar,
                                  std::uint64_t>(context, node, params);
  };
  return kTfLiteError;
}

template <KernelType kernel_type, typename SrcScalar>
TfLiteStatus EvalChooseOutputType(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteBConv2DParams* params) {
  const TfLiteType output_type = GetOutput(context, node, 0)->type;
  if (output_type == kTfLiteFloat32) {
    return EvalChooseBitpackType<kernel_type, SrcScalar, float>(context, node,
                                                                params);
  } else if (output_type == kTfLiteInt8) {
    return EvalChooseBitpackType<kernel_type, SrcScalar, std::int8_t>(
        context, node, params);
  } else if (params->write_bitpacked_output && output_type == kTfLiteInt32) {
    return EvalChooseBitpackType<kernel_type, SrcScalar, std::int32_t>(
        context, node, params);
  }
  return kTfLiteError;
}

template <KernelType kernel_type>
TfLiteStatus EvalChooseInputType(TfLiteContext* context, TfLiteNode* node,
                                 TfLiteBConv2DParams* params) {
  const TfLiteType input_type = GetInput(context, node, 0)->type;
  if (input_type == kTfLiteFloat32) {
    return EvalChooseOutputType<kernel_type, float>(context, node, params);
  } else if (input_type == kTfLiteInt8) {
    return EvalChooseOutputType<kernel_type, std::int8_t>(context, node,
                                                          params);
  } else if (params->read_bitpacked_input && input_type == kTfLiteInt32) {
    return EvalChooseOutputType<kernel_type, std::int32_t>(context, node,
                                                           params);
  }
  return kTfLiteError;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);
  return EvalChooseInputType<kernel_type>(context, node, params);
}

}  // namespace bconv2d

TfLiteRegistration* Register_BCONV_2D32_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericRef, 32>,
      bconv2d::Eval<bconv2d::KernelType::kGenericRef>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D32_OPT() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kRuyOptimized, 32>,
      bconv2d::Eval<bconv2d::KernelType::kRuyOptimized>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D64_OPT() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kRuyOptimized, 64>,
      bconv2d::Eval<bconv2d::KernelType::kRuyOptimized>};
  return &r;
}

// Use this registration wrapper to decide which impl. to use.
TfLiteRegistration* Register_BCONV_2D() {
#if defined TFLITE_WITH_RUY

#if RUY_PLATFORM_ARM_32
  return Register_BCONV_2D32_OPT();
#else  // ARM 64 and x86
  return Register_BCONV_2D64_OPT();
#endif

#else  // disabled TFLITE_WITH_RUY

  // When the RUY is disabled, we run the 32-bit reference implementation
  // on both 32-bit and 64-bit architectures.
  return Register_BCONV_2D32_REF();

#endif  // defined TFLITE_WITH_RUY
}

}  // namespace tflite
}  // namespace compute_engine
