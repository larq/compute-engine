#include <cstdint>

#include "bconv2d_impl.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/bconv2d_functor.h"
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
  // kReference: the same impl. path as the BConv2D op for TF
  kReference,
  // kGenericOptimized: the impl. path using optimized BGEMM kernel
  kGenericOptimized,
  // kRuyOptimized: the impl. path using RUY framework
  kRuyOptimized,  // TODO
};

const int kTensorNotAllocated = -1;

typedef struct {
  // input tensor dimensions
  int64_t batch{0};
  int64_t input_width{0};
  int64_t input_height{0};

  // filters tensor dimensions
  int64_t filter_width{0};
  int64_t filter_height{0};
  int64_t channels_in{0};
  int64_t channels_out{0};

  // strides
  int64_t strides[4] = {};

  // dilations
  int64_t dilations[4] = {};

  // padding
  TfLitePadding padding_type{};
  TfLitePaddingValues padding_values{};

  // output tensor dimensions
  int64_t out_width{0};
  int64_t out_height{0};

  ce::core::FilterFormat filter_format{ce::core::FilterFormat::Unknown};

  bool bitpack_before_im2col = false;
  bool need_im2col = false;
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers. They are unique in the entire TF Lite context.
  int im2col_id = kTensorNotAllocated;
  // In node->temporaries there is a list of tensor id's that are part
  // of this node in particular. The indices below are offsets into this array.
  // So in pseudo-code: `node->temporaries[index] = id;`
  int32_t im2col_index;

  std::vector<std::uint8_t> padding_buffer;
  std::vector<std::uint8_t> bitpacked_weights_buffer;

  bool is_weight_bitpacked = false;
  bool is_padding_correction_cached = false;

} TfLiteBConv2DParams;

inline void decide_bitpack_before_im2col(TfLiteBConv2DParams* conv_params,
                                         const int bitwidth) {
  if (conv_params->channels_in >= bitwidth / 4) {
    conv_params->bitpack_before_im2col = true;
  } else {
    conv_params->bitpack_before_im2col = false;
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* conv_params = new TfLiteBConv2DParams{};

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  // Later we can change this so that we only allow "OHWI_PACKED" (prepacked)
  if (m["filter_format"].IsNull() || m["filter_format"].ToString() == "HWIO") {
    conv_params->filter_format = ce::core::FilterFormat::HWIO;
  } else if (m["filter_format"].ToString() == "OHWI") {
    conv_params->filter_format = ce::core::FilterFormat::OHWI;
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
  conv_params->padding_type =
      m["padding"].ToString() == "VALID" ||
              m["padding"].ToString() ==
                  "valid"  // TODO: not sure if this check is needed
          ? TfLitePadding::kTfLitePaddingValid
          : TfLitePadding::kTfLitePaddingSame;
  return conv_params;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteBConv2DParams*>(buffer);
}

TfLiteStatus Prepare(KernelType kernel_type, const int bitwidth,
                     TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size == 4);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* fused_multiply = GetInput(context, node, 2);
  const auto* fused_add = GetInput(context, node, 3);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(fused_multiply), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(fused_add), 1);

  // The inputs fused_mutiply and fused_add are currently float
  // in order to accomodate for batchnorm scales
  // Later this might be changed to the int8 system of multipliers+shifts
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, fused_multiply->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, fused_add->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  // reading the input dimensions
  // TF and TF lite have the same input format [B, H, W, Ci]
  conv_params->batch = input->dims->data[0];
  conv_params->input_height = input->dims->data[1];
  conv_params->input_width = input->dims->data[2];
  conv_params->channels_in = input->dims->data[3];

  // reading the filter dimensions
  // only OHWI layout is supported for filters
  TF_LITE_ENSURE_EQ(context, conv_params->filter_format,
                    ce::core::FilterFormat::OHWI);

  conv_params->channels_out = filter->dims->data[0];
  conv_params->filter_height = filter->dims->data[1];
  conv_params->filter_width = filter->dims->data[2];
  TF_LITE_ENSURE_EQ(context, conv_params->channels_in, filter->dims->data[3]);

  TF_LITE_ENSURE_EQ(context, fused_multiply->dims->data[0],
                    conv_params->channels_out);
  TF_LITE_ENSURE_EQ(context, fused_add->dims->data[0],
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
  decide_bitpack_before_im2col(conv_params, bitwidth);

  // pre-allocate temporary tensors for optimized version
  if (kernel_type == KernelType::kGenericOptimized) {
    conv_params->need_im2col =
        (conv_params->strides[2] /* width */ != 1 ||
         conv_params->strides[1] /* height */ != 1 ||
         conv_params->dilations[2] /* width */ != 1 ||
         conv_params->dilations[1] /* height */ != 1 ||
         conv_params->filter_width != 1 || conv_params->filter_height != 1);

    if (conv_params->need_im2col) {
      conv_params->im2col_index = 0;

      // Allocate int array of that size
      TfLiteIntArrayFree(node->temporaries);
      node->temporaries = TfLiteIntArrayCreate(1);

      // Now allocate the buffers
      if (conv_params->im2col_id == kTensorNotAllocated) {
        context->AddTensors(context, 1, &conv_params->im2col_id);
        node->temporaries->data[conv_params->im2col_index] =
            conv_params->im2col_id;
      }

      // Resize the tensor
      int channels_in =
          conv_params->bitpack_before_im2col
              ? ((conv_params->channels_in + bitwidth - 1) / bitwidth)
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
        if (bitwidth == 8)
          im2col->type = kTfLiteInt8;
        else if (bitwidth == 32)
          im2col->type = kTfLiteInt32;
        else if (bitwidth == 64)
          im2col->type = kTfLiteInt64;
        else
          TF_LITE_ENSURE(context, false);
      } else {
        // im2col before bitpacking so use the same type as the input
        im2col->type = input->type;
      }
      im2col->allocation_type = kTfLiteArenaRw;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, im2col, im2col_size));
    }
  }

  conv_params->is_weight_bitpacked = false;
  conv_params->is_padding_correction_cached = false;

  return kTfLiteOk;
}

template <KernelType kernel_type, int bitwidth>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, bitwidth, context, node);
}

template <class T, class TBitpacked>
void EvalRef(TfLiteContext* context, TfLiteNode* node,
             const TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  const auto stride_height = params->strides[1];
  const auto stride_width = params->strides[2];
  const int padding =
      params->padding_type == TfLitePadding::kTfLitePaddingValid ? 1 : 2;

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, Layout::RowMajor, TBitpacked,
                                      Layout::ColMajor, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, Layout::RowMajor, T, Layout::ColMajor, T,
                                  TBitpacked, TBGemmFunctor>;
  using TConvFunctor =
      ce::core::Im2ColBConvFunctor<T, T, T, TFusedBGemmFunctor>;
  using PaddingFunctor =
      ce::core::PaddingFunctor<T, T, ce::core::FilterFormat::OHWI>;

  static TConvFunctor conv_functor;
  conv_functor(input->data.f, params->batch, params->input_height,
               params->input_width, params->channels_in, filter->data.f,
               params->filter_height, params->filter_width,
               params->channels_out, stride_height, stride_width, padding,
               output->data.f, params->out_height, params->out_width);

  if (params->padding_type == TfLitePadding::kTfLitePaddingSame) {
    PaddingFunctor padding_functor;
    padding_functor(params->batch, params->input_height, params->input_width,
                    params->channels_in, filter->data.f, params->filter_height,
                    params->filter_width, params->channels_out, stride_height,
                    stride_width, params->dilations[1], params->dilations[2],
                    output->data.f, params->out_height, params->out_width);
  }
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

  // TODO: this is not required for binary conv, however we need to
  // check if it is used in other components of TF lite internal method which
  // we are reusing and what are the default values!
  // op_params.float_activation_min = output_activation_min;
  // op_params.float_activation_max = output_activation_max;
}

template <class T, class TBitpacked>
void EvalOpt(TfLiteContext* context, TfLiteNode* node,
             TfLiteBConv2DParams* params) {
  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  const auto* fused_multiply = GetInput(context, node, 2);
  const auto* fused_add = GetInput(context, node, 3);
  auto* output = GetOutput(context, node, 0);

  TfLiteTensor* im2col = params->need_im2col
                             ? GetTemporary(context, node, params->im2col_index)
                             : nullptr;

  // Only in the first run:
  // Allocate the padding buffer and compute correction values
  if (params->padding_type == TfLitePadding::kTfLitePaddingSame &&
      !params->is_padding_correction_cached) {
    using PaddingFunctor =
        ce::core::PaddingFunctor<float, float, ce::core::FilterFormat::OHWI>;
    PaddingFunctor padding_functor;

    std::size_t padding_cache_size =
        padding_functor.get_cache_size(
            params->filter_height, params->filter_width, params->channels_out,
            params->dilations[1], params->dilations[2]) *
        sizeof(float);

    params->padding_buffer.resize(padding_cache_size);

    padding_functor.cache_correction_values(
        GetTensorData<float>(filter), params->filter_height,
        params->filter_width, params->channels_out, params->channels_in,
        params->dilations[1], params->dilations[2],
        GetTensorData<float>(fused_multiply),
        reinterpret_cast<float*>(params->padding_buffer.data()));
    params->is_padding_correction_cached = true;
  }

  // Only in the first run:
  // Allocate the packed weight buffer and bitpack the weights.
  // Ideally we would like to use the filter buffer itself,
  // but this is stored in read-only memory-mapped-files..
  if (!params->is_weight_bitpacked) {
    // The filters have shape
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
    size_t filter_rows_bp, filter_cols_bp, filter_bitpadding;
    ce::core::packbits_matrix(GetTensorData<float>(filter), rows, cols,
                              filter_data_bp, filter_rows_bp, filter_cols_bp,
                              filter_bitpadding, ce::core::Axis::RowWise);

    size_t num_bytes = filter_data_bp.size() * sizeof(TBitpacked);

    params->bitpacked_weights_buffer.resize(num_bytes);
    memcpy(params->bitpacked_weights_buffer.data(), filter_data_bp.data(),
           num_bytes);

    params->is_weight_bitpacked = true;
  }

  // Using the standard TF Lite ConvParams struct.
  // This requires extra step of converting the TfLiteBConv2DParams
  // but unifies the interface with the default TF lite API for CONV params
  // which is used in internal TF lite im2col functions.
  ConvParams op_params;
  GetConvParamsType(*params, op_params);

  // We pass the shape of the original unpacked filter, so that all the shape
  // information is correct (number of channels etc), but we pass the packed
  // weights data
  BConv2D<T, TBitpacked>(
      op_params, GetTensorShape(input), GetTensorData<T>(input),
      GetTensorShape(filter),
      reinterpret_cast<TBitpacked*>(params->bitpacked_weights_buffer.data()),
      GetTensorData<float>(fused_multiply), GetTensorData<float>(fused_add),
      GetTensorShape(output), GetTensorData<T>(output), GetTensorShape(im2col),
      GetTensorData<T>(im2col), params->bitpack_before_im2col,
      reinterpret_cast<T*>(params->padding_buffer.data()),
      CpuBackendContext::GetFromContext(context));
}

template <KernelType kernel_type, class TBitpacked>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  if (kernel_type == KernelType::kReference) {
    EvalRef<float, TBitpacked>(context, node, conv_params);
  } else if (kernel_type == KernelType::kGenericOptimized) {
    EvalOpt<float, TBitpacked>(context, node, conv_params);
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace bconv2d

// TfLiteRegistration* Register_BCONV_2D8_REF() {
//   static TfLiteRegistration r = {
//       bconv2d::Init, bconv2d::Free,
//       bconv2d::Prepare<bconv2d::KernelType::kReference, 8>,
//       bconv2d::Eval<bconv2d::KernelType::kReference, std::uint8_t>};
//   return &r;
// }

TfLiteRegistration* Register_BCONV_2D32_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference, 32>,
      bconv2d::Eval<bconv2d::KernelType::kReference, std::uint32_t>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D64_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference, 64>,
      bconv2d::Eval<bconv2d::KernelType::kReference, std::uint64_t>};
  return &r;
}

// TfLiteRegistration* Register_BCONV_2D8_OPT() {
//   static TfLiteRegistration r = {
//       bconv2d::Init, bconv2d::Free,
//       bconv2d::Prepare<bconv2d::KernelType::kGenericOptimized, 8>,
//       bconv2d::Eval<bconv2d::KernelType::kGenericOptimized, std::uint8_t>};
//   return &r;
// }

TfLiteRegistration* Register_BCONV_2D32_OPT() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericOptimized, 32>,
      bconv2d::Eval<bconv2d::KernelType::kGenericOptimized, std::uint32_t>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D64_OPT() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kGenericOptimized, 64>,
      bconv2d::Eval<bconv2d::KernelType::kGenericOptimized, std::uint64_t>};
  return &r;
}

// use these registration wrappers to decide which impl. to use.
// TfLiteRegistration* Register_BCONV_2D8() {
// #if defined TFLITE_WITH_RUY
//   return Register_BCONV_2D8_OPT();
// #else
//   return Register_BCONV_2D8_REF();
// #endif
// }

TfLiteRegistration* Register_BCONV_2D32() {
#if defined TFLITE_WITH_RUY
  return Register_BCONV_2D32_OPT();
#else
  return Register_BCONV_2D32_REF();
#endif
}

TfLiteRegistration* Register_BCONV_2D64() {
#if defined TFLITE_WITH_RUY
  return Register_BCONV_2D64_OPT();
#else
  return Register_BCONV_2D64_REF();
#endif
}

}  // namespace tflite
}  // namespace compute_engine
