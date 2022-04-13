#include <cmath>
#include <cstdint>

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/core/bconv2d/optimized_bgemm.h"
#include "larq_compute_engine/core/bconv2d/optimized_indirect_bgemm.h"
#include "larq_compute_engine/core/bconv2d/params.h"
#include "larq_compute_engine/core/bconv2d/reference.h"
#include "larq_compute_engine/core/bconv2d/zero_padding_correction.h"
#include "larq_compute_engine/core/indirect_bgemm/kernel.h"
#include "larq_compute_engine/core/indirect_bgemm/select_kernel.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/kernels/utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {
namespace bconv2d {

using core::TBitpacked;
using namespace core::bconv2d;
using namespace core::bitpacking;

enum class KernelType {
  // The reference implementation with for-loops.
  kReference,

  // The Ruy implementation with im2col and hand-optimized BGEMM kernels.
  kOptimizedBGEMM,

  // The XNNPack-derived implementation with indirect BGEMM kernels.
  kOptimizedIndirectBGEMM,
};

constexpr int kTensorNotAllocated = -1;

struct OpData {
  BConv2DParams params;

  // Fused activation function
  TfLiteFusedActivation fused_activation_function;

  // Computed output transform values. These are only used when writing
  // float/int8 output.
  std::int32_t output_transform_clamp_min;
  std::int32_t output_transform_clamp_max;
  std::vector<float> output_transform_multiplier;
  std::vector<float> output_transform_bias;

  // This is used when we have 'same-zero' padding.
  std::vector<float> padding_buffer;

  // This is used for the 'indirect bgemm' kernel type.
  std::unique_ptr<core::indirect_bgemm::Kernel> indirect_bgemm_kernel;

  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers. They are unique in the entire TF Lite context.
  int im2col_id = kTensorNotAllocated;
  // In node->temporaries there is a list of tensor id's that are part
  // of this node in particular. The indices below are offsets into this array.
  // So in pseudo-code: `node->temporaries[index] = id;`
  std::int32_t im2col_index = -1;

  bool successfully_initialized = false;

  bool one_time_setup_complete = false;
};

#define LCE_ENSURE_PARAM(op_data, context, a)                           \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return op_data;                                                   \
    }                                                                   \
  } while (0)

void* Init(TfLiteContext* context, const char* buffer, std::size_t length) {
  auto* op_data = new OpData{};
  auto* bconv2d_params = &op_data->params;

  const std::uint8_t* buffer_t = reinterpret_cast<const std::uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  // Read the op's input arguments into the `bconv2d_params` struct

  LCE_ENSURE_PARAM(op_data, context, !m["stride_height"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["stride_width"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["dilation_height_factor"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["dilation_width_factor"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["padding"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["pad_values"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["channels_in"].IsNull());
  LCE_ENSURE_PARAM(op_data, context, !m["fused_activation_function"].IsNull());

  bconv2d_params->stride_height = m["stride_height"].AsInt32();
  bconv2d_params->stride_width = m["stride_width"].AsInt32();

  bconv2d_params->dilation_height_factor =
      m["dilation_height_factor"].AsInt32();
  bconv2d_params->dilation_width_factor = m["dilation_width_factor"].AsInt32();

  bconv2d_params->padding_type =
      ConvertPadding((Padding)m["padding"].AsInt32());
  bconv2d_params->pad_value = m["pad_values"].AsInt32();
  if (bconv2d_params->pad_value != 0 && bconv2d_params->pad_value != 1) {
    TF_LITE_KERNEL_LOG(context, "Attribute pad_values must be 0 or 1.");
    return op_data;
  }

  // The input and filters are bitpacked along the channel in axis, which means
  // we cannot infer the 'true' input shape, so there's an explicit integer
  // attribute added to the op in the converter.
  bconv2d_params->channels_in = m["channels_in"].AsInt32();

  op_data->fused_activation_function = ConvertActivation(
      (ActivationFunctionType)m["fused_activation_function"].AsInt32());

  // It's not possible to return an error code in this method. If we get to here
  // without returning early, initialisation has succeeded without error, and so
  // we set this flag. If it's unset in Prepare, we report the error there.
  op_data->successfully_initialized = true;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  auto* bconv2d_params = &op_data->params;

  // If an error happened in Init, then return an error code.
  if (!op_data->successfully_initialized) return kTfLiteError;

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
  bconv2d_params->channels_out = SizeOfDimension(filter, 0);
  bconv2d_params->filter_height = SizeOfDimension(filter, 1);
  bconv2d_params->filter_width = SizeOfDimension(filter, 2);

  if (SizeOfDimension(filter, 3) ==
      GetBitpackedSize(bconv2d_params->channels_in)) {
    bconv2d_params->groups = 1;
  } else {
    TF_LITE_ENSURE_MSG(
        context, kernel_type != KernelType::kOptimizedBGEMM,
        "Grouped binary convolutions are not supported with this kernel.");
    TF_LITE_ENSURE_EQ(context,
                      GetBitpackedSize(bconv2d_params->channels_in) %
                          SizeOfDimension(filter, 3),
                      0);
    const std::int32_t groups = GetBitpackedSize(bconv2d_params->channels_in) /
                                SizeOfDimension(filter, 3);
    const std::int32_t group_size = bconv2d_params->channels_in / groups;
    TF_LITE_ENSURE_EQ(context, group_size % core::bitpacking_bitwidth, 0);
    TF_LITE_ENSURE_EQ(context, bconv2d_params->channels_out % groups, 0);
    bconv2d_params->groups = groups;
  }

  if (bconv2d_params->padding_type == kTfLitePaddingSame &&
      bconv2d_params->pad_value == 0) {
    TF_LITE_ENSURE_MSG(
        context,
        (kernel_type == KernelType::kReference &&
         bconv2d_params->channels_in % 2 == 0) ||
            (kernel_type != KernelType::kReference &&
             output->type == kTfLiteFloat32 &&
             op_data->fused_activation_function == kTfLiteActNone),
        "Zero-padding is only supported by the reference kernel with an even "
        "number of input channels, or when using "
        "float output with no fused activation function.");
  }

  // Compute the padding and output values (height, width)
  int out_width, out_height;
  bconv2d_params->padding_values = ComputePaddingHeightWidth(
      bconv2d_params->stride_height, bconv2d_params->stride_width,
      bconv2d_params->dilation_height_factor,
      bconv2d_params->dilation_width_factor, SizeOfDimension(input, 1),
      SizeOfDimension(input, 2), bconv2d_params->filter_height,
      bconv2d_params->filter_width, bconv2d_params->padding_type, &out_height,
      &out_width);

  if (output->type == kTfLiteInt32) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(thresholds), 1);
    TF_LITE_ENSURE_EQ(context, thresholds->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(thresholds, 0),
                      bconv2d_params->channels_out);
  } else {
    TF_LITE_ENSURE_EQ(context, post_activation_multiplier->type,
                      kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, post_activation_bias->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_multiplier), 1);
    TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_bias), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(post_activation_multiplier, 0),
                      bconv2d_params->channels_out);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(post_activation_bias, 0),
                      bconv2d_params->channels_out);
  }

  if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
  }

  if (kernel_type == KernelType::kOptimizedIndirectBGEMM) {
    TF_LITE_ENSURE_MSG(
        context, input->allocation_type != kTfLiteDynamic,
        "The input tensor must not have dynamic allocation type");
  }

  // Determine the output dimensions and allocate the output buffer
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(4);
  output_shape->data[0] = SizeOfDimension(input, 0);
  output_shape->data[1] = out_height;
  output_shape->data[2] = out_width;
  output_shape->data[3] = output->type == kTfLiteInt32
                              ? GetBitpackedSize(bconv2d_params->channels_out)
                              : bconv2d_params->channels_out;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // Figure out how many temporary buffers we need
  int temporaries_count = 0;

  const bool need_im2col =
      kernel_type == KernelType::kOptimizedBGEMM &&
      (bconv2d_params->stride_width != 1 ||
       bconv2d_params->stride_height != 1 ||
       bconv2d_params->dilation_width_factor != 1 ||
       bconv2d_params->dilation_height_factor != 1 ||
       bconv2d_params->filter_width != 1 || bconv2d_params->filter_height != 1);

  // Pre-allocate temporary tensors
  if (need_im2col) {
    op_data->im2col_index = temporaries_count++;
  } else {
    op_data->im2col_index = -1;
  }

  if (temporaries_count != 0) {
    // Allocate int array of that size
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);
  }

  // Now allocate the buffers
  if (need_im2col) {
    // Allocate the im2col tensor if necessary
    if (op_data->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &op_data->im2col_id);
    }
    node->temporaries->data[op_data->im2col_index] = op_data->im2col_id;

    // Resize the im2col tensor
    const std::int32_t bitpacked_channels_in =
        GetBitpackedSize(bconv2d_params->channels_in);
    TfLiteIntArray* im2col_size = TfLiteIntArrayCopy(output_shape);
    im2col_size->data[3] = bitpacked_channels_in *
                           bconv2d_params->filter_height *
                           bconv2d_params->filter_width;
    TfLiteTensor* im2col = GetTemporary(context, node, op_data->im2col_index);
    im2col->type = kTfLiteInt32;
    im2col->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, im2col, im2col_size));
  }

  // Prepare could be called multiple times; when the input tensor is resized,
  // we shoud always re-do the one-time setup.
  op_data->one_time_setup_complete = false;

  return kTfLiteOk;
}

inline void GetConvParamsType(const OpData* op_data,
                              ConvParams& conv2d_params) {
  const auto* bconv2d_params = &op_data->params;

  // Padding
  conv2d_params.padding_type = RuntimePaddingType(bconv2d_params->padding_type);
  conv2d_params.padding_values.width = bconv2d_params->padding_values.width;
  conv2d_params.padding_values.height = bconv2d_params->padding_values.height;

  // Strides
  conv2d_params.stride_height = bconv2d_params->stride_height;
  conv2d_params.stride_width = bconv2d_params->stride_width;

  // Dilations
  conv2d_params.dilation_height_factor = bconv2d_params->dilation_height_factor;
  conv2d_params.dilation_width_factor = bconv2d_params->dilation_width_factor;

  // Activation function
  conv2d_params.quantized_activation_min = op_data->output_transform_clamp_min;
  conv2d_params.quantized_activation_max = op_data->output_transform_clamp_max;
}

void OneTimeSetup(TfLiteContext* context, TfLiteNode* node, OpData* op_data) {
  const auto* bconv2d_params = &op_data->params;

  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  const auto* output = GetOutput(context, node, 0);

  // Division is safe because at this point we know that channels_in is a
  // multiple of the number of groups.
  const std::int32_t channels_in_per_group =
      bconv2d_params->channels_in / bconv2d_params->groups;

  // For 'same-zero' padding, compute the padding-correction.
  if (bconv2d_params->padding_type == kTfLitePaddingSame &&
      bconv2d_params->pad_value == 0) {
    op_data->padding_buffer.resize(zero_padding_correction::GetCacheSize(
        bconv2d_params->filter_height, bconv2d_params->filter_width,
        bconv2d_params->channels_out, bconv2d_params->dilation_height_factor,
        bconv2d_params->dilation_width_factor));
    zero_padding_correction::CacheCorrectionValues(
        GetTensorData<TBitpacked>(filter), bconv2d_params->filter_height,
        bconv2d_params->filter_width, bconv2d_params->channels_out,
        channels_in_per_group, bconv2d_params->dilation_height_factor,
        bconv2d_params->dilation_width_factor,
        GetTensorData<float>(post_activation_multiplier),
        op_data->padding_buffer.data());
  }

  // If applicable, fuse the back-transformation and int8 scale/zero-point into
  // the output transform multiplier/bias.
  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt8) {
    op_data->output_transform_multiplier.resize(
        bconv2d_params->channels_out + LCE_EXTRA_BYTES / sizeof(float));
    op_data->output_transform_bias.resize(bconv2d_params->channels_out +
                                          LCE_EXTRA_BYTES / sizeof(float));

    const auto filter_shape = GetTensorShape(GetInput(context, node, 1));
    const std::int32_t backtransform_add =
        filter_shape.Dims(1) * filter_shape.Dims(2) * channels_in_per_group;
    const double output_scale =
        output->type == kTfLiteInt8 ? output->params.scale : 1.0;
    const double output_zero_point =
        output->type == kTfLiteInt8 ? output->params.zero_point : 0.0;

    for (int i = 0; i < bconv2d_params->channels_out; ++i) {
      const double post_mul =
          GetTensorData<float>(post_activation_multiplier)[i];
      const double post_bias = GetTensorData<float>(post_activation_bias)[i];
      op_data->output_transform_multiplier.at(i) = -1 * post_mul / output_scale;
      op_data->output_transform_bias.at(i) =
          (post_bias + static_cast<double>(backtransform_add) * post_mul) /
              output_scale +
          output_zero_point;
    }

    std::int32_t nominal_clamp_min, nominal_clamp_max;
    CalculateActivationRange(op_data->fused_activation_function,
                             &nominal_clamp_min, &nominal_clamp_max);
    nominal_clamp_min = std::max(nominal_clamp_min, -1 * backtransform_add);
    nominal_clamp_max = std::min(nominal_clamp_max, backtransform_add);
    op_data->output_transform_clamp_min =
        -1 * nominal_clamp_max + backtransform_add;
    op_data->output_transform_clamp_max =
        -1 * nominal_clamp_min + backtransform_add;
  }

  op_data->one_time_setup_complete = true;
}

// Fill in the OutputTransform values for float and/or int8 outputs
template <typename DstScalar>
void GetOutputTransform(OutputTransform<DstScalar>& output_transform,
                        TfLiteContext* context, TfLiteNode* node,
                        OpData* op_data) {
  static_assert(std::is_same<DstScalar, float>::value ||
                    std::is_same<DstScalar, std::int8_t>::value,
                "");
  output_transform.clamp_min = op_data->output_transform_clamp_min;
  output_transform.clamp_max = op_data->output_transform_clamp_max;
  output_transform.multiplier = op_data->output_transform_multiplier.data();
  output_transform.bias = op_data->output_transform_bias.data();
}

// Fill in the OutputTransform values for bitpacked outputs
void GetOutputTransform(OutputTransform<TBitpacked>& output_transform,
                        TfLiteContext* context, TfLiteNode* node,
                        OpData* op_data) {
  const auto* thresholds = GetInput(context, node, 4);
  output_transform.thresholds = GetTensorData<std::int32_t>(thresholds);
}

template <typename AccumScalar, typename DstScalar>
void EvalOptBGEMM(TfLiteContext* context, TfLiteNode* node, OpData* op_data) {
  if (!op_data->one_time_setup_complete) {
    OneTimeSetup(context, node, op_data);
  }

  auto* bconv2d_params = &op_data->params;

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TfLiteTensor* im2col =
      op_data->im2col_index >= 0
          ? GetTemporary(context, node, op_data->im2col_index)
          : nullptr;

  // Using the standard TF Lite ConvParams struct. This requires extra step of
  // converting the BConv2DParams but unifies the interface with the default
  // TFLite API for Conv2D params which is used in internal TFLite im2col
  // functions.
  ConvParams conv2d_params;
  GetConvParamsType(op_data, conv2d_params);
  conv2d_params.input_offset = input->params.zero_point;

  OutputTransform<DstScalar> output_transform;
  GetOutputTransform(output_transform, context, node, op_data);

  // `BConv2D` wants the *unpacked* output shape.
  auto unpacked_output_shape = GetTensorShape(output);
  unpacked_output_shape.SetDim(3, bconv2d_params->channels_out);

  // We pass the shape of the original unpacked filter, so that all the shape
  // information is correct (number of channels etc), but we pass the packed
  // weights data.
  //     Likewise, we pass the original output shape even if we are going to
  // write bitpacked output directly.
  BConv2DOptimizedBGEMM<AccumScalar, DstScalar>(
      conv2d_params, GetTensorShape(input), GetTensorData<TBitpacked>(input),
      GetTensorShape(filter), GetTensorData<TBitpacked>(filter),
      output_transform, unpacked_output_shape, GetTensorData<DstScalar>(output),
      GetTensorShape(im2col), GetTensorData<TBitpacked>(im2col),
      op_data->padding_buffer.data(), bconv2d_params->pad_value,
      CpuBackendContext::GetFromContext(context));
}

template <typename DstScalar>
void EvalOptIndirectBGEMM(TfLiteContext* context, TfLiteNode* node,
                          OpData* op_data) {
  if (!op_data->one_time_setup_complete) {
    OneTimeSetup(context, node, op_data);
  }

  auto* bconv2d_params = &op_data->params;

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  const auto bitpacked_input_shape = GetTensorShape(input);
  const auto output_shape = GetTensorShape(output);

  if (!op_data->indirect_bgemm_kernel) {
    OutputTransform<DstScalar> output_transform;
    GetOutputTransform(output_transform, context, node, op_data);
    op_data->indirect_bgemm_kernel =
        std::move(core::indirect_bgemm::SelectRuntimeKernel(
            &op_data->params, bitpacked_input_shape, output_shape,
            output_transform));
    op_data->indirect_bgemm_kernel->PackWeights(
        GetTensorData<TBitpacked>(filter));
    op_data->indirect_bgemm_kernel->FillIndirectionBuffer(
        &op_data->params, bitpacked_input_shape, output_shape,
        GetTensorData<TBitpacked>(input));
  }

  BConv2DOptimizedIndirectBGEMM<DstScalar>(
      op_data->indirect_bgemm_kernel.get(), bconv2d_params,
      bitpacked_input_shape, output_shape, GetTensorData<DstScalar>(output),
      op_data->padding_buffer.data(), bconv2d_params->pad_value);
}

template <typename DstScalar>
void EvalRef(TfLiteContext* context, TfLiteNode* node, OpData* op_data) {
  if (!op_data->one_time_setup_complete) {
    OneTimeSetup(context, node, op_data);
  }

  const auto* input = GetInput(context, node, 0);
  const auto* packed_filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  OutputTransform<DstScalar> output_transform;
  GetOutputTransform(output_transform, context, node, op_data);

  BConv2DReference<std::int32_t, DstScalar>(
      &op_data->params, GetTensorShape(input), GetTensorData<TBitpacked>(input),
      GetTensorShape(packed_filter), GetTensorData<TBitpacked>(packed_filter),
      output_transform, GetTensorShape(output),
      GetTensorData<DstScalar>(output));
}

template <KernelType kernel_type, typename DstScalar>
inline TfLiteStatus EvalChooseKernelType(TfLiteContext* context,
                                         TfLiteNode* node, OpData* op_data) {
  if (kernel_type == KernelType::kOptimizedBGEMM) {
#if RUY_PLATFORM_ARM_64
    // On 64 bit Arm only there is an optimised kernel with 16-bit accumulators.
    // It is safe to use this without risk of overflow as long as the maximum
    // value of the convolution (filter height * filter width * input channels,
    // plus some overhead to account for potential padding) is less than 2^16.
    // We will almost always take this path: for a 3x3 filter there would need
    // to be > 7000 input channels to present an overflow risk.
    const int depth = op_data->params.filter_height *
                      op_data->params.filter_width *
                      op_data->params.channels_in;
    if (depth + 512 < 1 << 16) {
      EvalOptBGEMM<std::int16_t, DstScalar>(context, node, op_data);
      return kTfLiteOk;
    }
#endif
    // In all other cases, use 32-bit accumulators.
    EvalOptBGEMM<std::int32_t, DstScalar>(context, node, op_data);
    return kTfLiteOk;
  } else if (kernel_type == KernelType::kOptimizedIndirectBGEMM) {
    EvalOptIndirectBGEMM<DstScalar>(context, node, op_data);
    return kTfLiteOk;
  } else if (kernel_type == KernelType::kReference) {
    EvalRef<DstScalar>(context, node, op_data);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteType output_type = GetOutput(context, node, 0)->type;
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  if (output_type == kTfLiteFloat32) {
    return EvalChooseKernelType<kernel_type, float>(context, node, op_data);
  } else if (output_type == kTfLiteInt8) {
    return EvalChooseKernelType<kernel_type, std::int8_t>(context, node,
                                                          op_data);
  } else if (output_type == kTfLiteInt32) {
    return EvalChooseKernelType<kernel_type, TBitpacked>(context, node,
                                                         op_data);
  }
  return kTfLiteError;
}

}  // namespace bconv2d

TfLiteRegistration* Register_BCONV_2D_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference>,
      bconv2d::Eval<bconv2d::KernelType::kReference>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D_OPT_BGEMM() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kOptimizedBGEMM>,
      bconv2d::Eval<bconv2d::KernelType::kOptimizedBGEMM>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D_OPT_INDIRECT_BGEMM() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kOptimizedIndirectBGEMM>,
      bconv2d::Eval<bconv2d::KernelType::kOptimizedIndirectBGEMM>};
  return &r;
}

// Use this registration wrapper to decide which implementation to use.
TfLiteRegistration* Register_BCONV_2D() {
#if defined TFLITE_WITH_RUY
  return Register_BCONV_2D_OPT_BGEMM();
#else
  return Register_BCONV_2D_REF();
#endif
}

}  // namespace tflite
}  // namespace compute_engine
