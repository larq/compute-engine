#include <cstdint>

#include "bconv2d_impl.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/bconv2d_impl_ref.h"
#include "larq_compute_engine/core/padding_functor.h"
#include "larq_compute_engine/core/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
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

const int kTensorNotAllocated = -1;

typedef struct {
  // input tensor dimensions
  std::int64_t batch{0};
  std::int64_t input_width{0};
  std::int64_t input_height{0};

  // filters tensor dimensions
  std::int64_t filter_width{0};
  std::int64_t filter_height{0};
  std::int64_t channels_in{0};
  std::int64_t channels_out{0};

  // strides
  std::int64_t strides[4] = {};

  // dilations
  std::int64_t dilations[4] = {};

  // padding
  TfLitePadding padding_type{};
  TfLitePaddingValues padding_values{};
  int pad_value = 0;  // Must be 0 or 1

  // output tensor dimensions
  std::int64_t out_width{0};
  std::int64_t out_height{0};

  ce::core::FilterFormat filter_format{ce::core::FilterFormat::Unknown};

  TfLiteFusedActivation activation = kTfLiteActNone;
  // These min,max take care of a Relu.
  // Later they will *also* do the clamping in order to go from int32 to int8
  std::int32_t output_activation_min;
  std::int32_t output_activation_max;

  bool bitpack_before_im2col = false;
  bool need_im2col = false;
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers. They are unique in the entire TF Lite context.
  int im2col_id = kTensorNotAllocated;
  // In node->temporaries there is a list of tensor id's that are part
  // of this node in particular. The indices below are offsets into this array.
  // So in pseudo-code: `node->temporaries[index] = id;`
  std::int32_t im2col_index;

  std::vector<float> padding_buffer;
  bool is_padding_correction_cached = false;

  // Weights in the flatbuffer file are bitpacked in a different
  // order than what is expected by the kernels, so we repack the weights
  std::vector<std::uint8_t> filter_packed;
  bool is_filter_repacked = false;

  int bitpacking_bitwidth;

  bool conv_params_initialized = false;
} TfLiteBConv2DParams;

inline void decide_bitpack_before_im2col(TfLiteBConv2DParams* conv_params) {
  if (conv_params->channels_in >= conv_params->bitpacking_bitwidth / 4) {
    conv_params->bitpack_before_im2col = true;
  } else {
    conv_params->bitpack_before_im2col = false;
  }
}

void* Init(TfLiteContext* context, const char* buffer, std::size_t length) {
  auto* conv_params = new TfLiteBConv2DParams{};

  const std::uint8_t* buffer_t = reinterpret_cast<const std::uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  // Later we can change this so that we only allow "OHWI_PACKED" (prepacked)
  if (m["filter_format"].IsNull() || m["filter_format"].ToString() == "HWIO") {
    conv_params->filter_format = ce::core::FilterFormat::HWIO;
  } else if (m["filter_format"].ToString() == "OHWI") {
    conv_params->filter_format = ce::core::FilterFormat::OHWI;
  } else if (m["filter_format"].ToString() == "OHWI_PACKED") {
    conv_params->filter_format = ce::core::FilterFormat::OHWI_PACKED;
  } else {
    context->ReportError(context, "Invalid filter format.");
    return conv_params;
  }

  // reading the op's input arguments into the "conv_params" struct
  // readng strides
  auto strides_vector = m["strides"].AsTypedVector();
  if (strides_vector.size() != 4) {
    context->ReportError(context, "Strides vector should have size 4.");
    return conv_params;
  }
  for (std::size_t i = 0; i < strides_vector.size(); ++i)
    conv_params->strides[i] = strides_vector[i].AsInt64();

  // reading dilations
  auto dilation_vector = m["dilations"].AsTypedVector();
  if (dilation_vector.size() != 4) {
    context->ReportError(context, "Dilations vector should have size 4.");
    return conv_params;
  }
  for (std::size_t i = 0; i < dilation_vector.size(); ++i)
    conv_params->dilations[i] = dilation_vector[i].AsInt64();

  // reading padding
  if (m["padding"].ToString() == "VALID" ||
      m["padding"].ToString() == "valid") {
    conv_params->padding_type = kTfLitePaddingValid;
  } else if (m["padding"].ToString() == "SAME" ||
             m["padding"].ToString() == "same") {
    conv_params->padding_type = kTfLitePaddingSame;
  } else {
    context->ReportError(context, "Invalid padding attribute.");
  }

  // Read fused activation
  if (m["activation"].IsNull() || m["activation"].ToString() == "" ||
      m["activation"].ToString() == "NONE") {
    conv_params->activation = kTfLiteActNone;
  } else if (m["activation"].ToString() == "RELU") {
    conv_params->activation = kTfLiteActRelu;
  } else if (m["activation"].ToString() == "RELU1") {
    conv_params->activation = kTfLiteActRelu1;
  } else if (m["activation"].ToString() == "RELU6") {
    conv_params->activation = kTfLiteActRelu6;
  } else {
    context->ReportError(context, "Invalid value for activation.");
    return conv_params;
  }

  conv_params->pad_value =
      m["pad_values"].IsNull() ? 0 : m["pad_values"].AsInt64();

  if (conv_params->pad_value != 0 && conv_params->pad_value != 1) {
    context->ReportError(context, "Attribute pad_values must be 0 or 1.");
    return conv_params;
  }

  if (conv_params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      conv_params->pad_value != 1 &&
      conv_params->activation != kTfLiteActNone) {
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

  TF_LITE_ENSURE(context, node->inputs->size == 4);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_multiplier), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(post_activation_bias), 1);

  // The inputs post_mutiply and post_activation_bias are currently float
  // in order to accomodate for batchnorm scales
  // Later this might be changed to the int8 system of multipliers+shifts
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, post_activation_multiplier->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, post_activation_bias->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  // TODO: more intelligent selection of the parameters `bitpacking_bitwidth`
  //       and `bitpack_before_im2col` based on benchmarking results
  //       (as in https://github.com/larq/compute-engine/issues/290).
  conv_params->bitpacking_bitwidth = default_bitpacking_bitwidth;

  // reading the input dimensions
  // TF and TF lite have the same input format [B, H, W, Ci]
  conv_params->batch = input->dims->data[0];
  conv_params->input_height = input->dims->data[1];
  conv_params->input_width = input->dims->data[2];
  conv_params->channels_in = input->dims->data[3];

  // reading the filter dimensions
  // only OHWI layout is supported for filters
  TF_LITE_ENSURE(
      context,
      conv_params->filter_format == ce::core::FilterFormat::OHWI ||
          conv_params->filter_format == ce::core::FilterFormat::OHWI_PACKED);

  conv_params->channels_out = filter->dims->data[0];
  conv_params->filter_height = filter->dims->data[1];
  conv_params->filter_width = filter->dims->data[2];
  if (conv_params->filter_format == ce::core::FilterFormat::OHWI) {
    TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, conv_params->channels_in, filter->dims->data[3]);
  } else {
    // TF Lite does not support the unsigned int32 type so we use int32 here
    TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt32);
  }

  TF_LITE_ENSURE_EQ(context, post_activation_multiplier->dims->data[0],
                    conv_params->channels_out);
  TF_LITE_ENSURE_EQ(context, post_activation_bias->dims->data[0],
                    conv_params->channels_out);

  // computing the padding and output values (height, width)
  int out_width, out_height;
  conv_params->padding_values = ComputePaddingHeightWidth(
      conv_params->strides[1] /* strides height */,
      conv_params->strides[2] /* strides width */,
      conv_params->dilations[1] /* dil. height */,
      conv_params->dilations[2] /* dil. width */, conv_params->input_height,
      conv_params->input_width, conv_params->filter_height,
      conv_params->filter_width, conv_params->padding_type, &out_height,
      &out_width);

  conv_params->out_width = out_width;
  conv_params->out_height = out_height;

  CalculateActivationRange(conv_params->activation,
                           &conv_params->output_activation_min,
                           &conv_params->output_activation_max);

  // determine the output dimensions
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(4);
  output_shape->data[0] = conv_params->batch;
  output_shape->data[1] = conv_params->out_height;
  output_shape->data[2] = conv_params->out_width;
  output_shape->data[3] = conv_params->channels_out;

  // allocate the output buffer
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));

  // Decide if we do bitpacking before or after im2col
  decide_bitpack_before_im2col(conv_params);

  // pre-allocate temporary tensors for optimized version
  if (kernel_type == KernelType::kRuyOptimized) {
    conv_params->need_im2col =
        (conv_params->strides[2] /* width */ != 1 ||
         conv_params->strides[1] /* height */ != 1 ||
         conv_params->dilations[2] /* width */ != 1 ||
         conv_params->dilations[1] /* height */ != 1 ||
         conv_params->filter_width != 1 || conv_params->filter_height != 1);

    // Figure out how many temporary buffers we need
    int temporaries_count = 0;
    if (conv_params->need_im2col) {
      conv_params->im2col_index = temporaries_count++;
    }

    // Allocate int array of that size
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);

    // Now allocate the buffers
    if (conv_params->need_im2col) {
      if (conv_params->im2col_id == kTensorNotAllocated) {
        context->AddTensors(context, 1, &conv_params->im2col_id);
        node->temporaries->data[conv_params->im2col_index] =
            conv_params->im2col_id;
      }
    }
  }

  // Resize the im2col tensor
  if (conv_params->need_im2col) {
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

  // Prepare could be called multiple times, when the input tensor is resized,
  // so we always reset these flags
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
  op_params.stride_height = conv_params.strides[1];
  op_params.stride_width = conv_params.strides[2];

  // dilations
  op_params.dilation_height_factor = conv_params.dilations[1];
  op_params.dilation_width_factor = conv_params.dilations[2];

  // Activation function
  op_params.quantized_activation_min = conv_params.output_activation_min;
  op_params.quantized_activation_max = conv_params.output_activation_max;
}

template <class T, class TBitpacked>
void EvalOpt(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  auto* output = GetOutput(context, node, 0);

  TfLiteTensor* im2col = params->need_im2col
                             ? GetTemporary(context, node, params->im2col_index)
                             : nullptr;

  if (!params->is_filter_repacked || !params->is_padding_correction_cached) {
    const std::uint32_t* filter_flatbuffer =
        GetTensorData<std::uint32_t>(filter);
    const T* filter_unpacked = nullptr;

    if (params->filter_format == ce::core::FilterFormat::OHWI_PACKED) {
      // First unpack the filter to float
      int cols = params->channels_in;
      int rows =
          params->channels_out * params->filter_height * params->filter_width;

      // This vector is declared static, so that it will be shared by all nodes.
      static std::vector<T> unpacked_weights;
      unpacked_weights.resize(rows * cols);

      ce::core::unpack_matrix(filter_flatbuffer, rows, cols,
                              unpacked_weights.data());

      filter_unpacked = unpacked_weights.data();
    } else {
      // Filter was already unpacked
      filter_unpacked = GetTensorData<T>(filter);
    }

    // Fill the zero-padding cache
    if (!params->is_padding_correction_cached &&
        (params->padding_type == TfLitePadding::kTfLitePaddingSame &&
         params->pad_value == 0)) {
      using PaddingFunctor =
          ce::core::PaddingFunctor<T, T, ce::core::FilterFormat::OHWI>;
      PaddingFunctor padding_functor;

      std::size_t padding_cache_size = padding_functor.get_cache_size(
          params->filter_height, params->filter_width, params->channels_out,
          params->dilations[1], params->dilations[2]);

      params->padding_buffer.resize(padding_cache_size);

      padding_functor.cache_correction_values(
          filter_unpacked, params->filter_height, params->filter_width,
          params->channels_out, params->channels_in, params->dilations[1],
          params->dilations[2], GetTensorData<T>(post_activation_multiplier),
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

    std::vector<TBitpacked> filter_data_bp;
    std::size_t filter_rows_bp, filter_cols_bp, filter_bitpadding;
    ce::core::packbits_matrix<ce::core::BitpackOrder::Optimized>(
        filter_unpacked, rows, cols, filter_data_bp, filter_rows_bp,
        filter_cols_bp, filter_bitpadding, ce::core::Axis::RowWise);

    std::size_t num_bytes = filter_data_bp.size() * sizeof(TBitpacked);

    params->filter_packed.resize(num_bytes);
    memcpy(params->filter_packed.data(), filter_data_bp.data(), num_bytes);

    params->is_filter_repacked = true;
  }

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);

  // `BConv2D` wants the *unpacked* filter shape
  auto unpacked_filter_shape = GetTensorShape(filter);
  unpacked_filter_shape.SetDim(3, GetTensorShape(input).Dims(3));

  // We pass the shape of the original unpacked filter, so that all the shape
  // information is correct (number of channels etc), but we pass the packed
  // weights data
  BConv2D<T, TBitpacked>(
      op_params, GetTensorShape(input), GetTensorData<T>(input),
      unpacked_filter_shape,
      reinterpret_cast<TBitpacked*>(params->filter_packed.data()),
      GetTensorData<float>(post_activation_multiplier),
      GetTensorData<float>(post_activation_bias), GetTensorShape(output),
      GetTensorData<T>(output), GetTensorShape(im2col),
      GetTensorData<T>(im2col), params->bitpack_before_im2col,
      params->padding_buffer.data(), params->pad_value,
      CpuBackendContext::GetFromContext(context));
}

template <class T, class TBitpacked>
void EvalRef(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* post_activation_multiplier = GetInput(context, node, 2);
  const auto* post_activation_bias = GetInput(context, node, 3);
  auto* output = GetOutput(context, node, 0);

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);

  TfLiteTensor* im2col = nullptr;
  ce::ref::BConv2D<T, TBitpacked>(
      op_params, GetTensorShape(input), GetTensorData<T>(input),
      GetTensorShape(filter),
      reinterpret_cast<TBitpacked*>(params->filter_packed.data()),
      GetTensorData<float>(post_activation_multiplier),
      GetTensorData<float>(post_activation_bias), GetTensorShape(output),
      GetTensorData<T>(output), GetTensorShape(im2col),
      GetTensorData<T>(im2col), false /*bitpack before im2col*/,
      nullptr /*padding buffer*/, params->pad_value,
      nullptr /*cpu backend context*/);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  if (kernel_type == KernelType::kRuyOptimized) {
    switch (conv_params->bitpacking_bitwidth) {
      case 32:
        EvalOpt<float, std::uint32_t>(context, node, conv_params);
        return kTfLiteOk;
      case 64:
        EvalOpt<float, std::uint64_t>(context, node, conv_params);
        return kTfLiteOk;
    }
  } else if (kernel_type == KernelType::kGenericRef) {
    switch (conv_params->bitpacking_bitwidth) {
      case 32:
        EvalRef<float, std::uint32_t>(context, node, conv_params);
        return kTfLiteOk;
      case 64:
        EvalRef<float, std::uint64_t>(context, node, conv_params);
        return kTfLiteOk;
    }
  }

  return kTfLiteError;
}
}  // namespace bconv2d

TfLiteRegistration* Register_BCONV_2D32_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericRef, 32>,
      bconv2d::Eval<bconv2d::KernelType::kGenericRef>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D64_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericRef, 64>,
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

#if RUY_PLATFORM(ARM_32)
  return Register_BCONV_2D32_OPT();
#else  // ARM 64 and x86
  return Register_BCONV_2D64_OPT();
#endif

#else  // disabled TFLITE_WITH_RUY

#if RUY_PLATFORM(ARM_32)
  return Register_BCONV_2D32_REF();
#else  // ARM 64 and x86
  return Register_BCONV_2D64_REF();
#endif

#endif
}

}  // namespace tflite
}  // namespace compute_engine
