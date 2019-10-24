#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

#include "larq_compute_engine/cc/core/bconv2d_functor.h"
#include "larq_compute_engine/cc/core/padding_functor.h"

#include <cstdint>

using namespace tflite;

namespace compute_engine {
namespace tflite {
namespace bconv2d {

namespace ce = compute_engine;

enum class KernelType {
  // kReference: the same code base as the reference implementation of BConv2D
  // op in TF
  kReference,
};

typedef struct {
  // TODO: double check the type of each variable
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
  TfLitePadding padding{};

  // output tensor dimensions
  int64_t out_width{0};
  int64_t out_height{0};

} TfLiteBConv2DParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* conv_params = new TfLiteBConv2DParams{};

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

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
  conv_params->padding =
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

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);

  // TF lite supports only single precision float as tensor data type!
  // Therefore no need to check against doubles for now.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  // reading the input dimensions
  // TF and TF lite have the same input format [B, H, W, Ci]
  conv_params->batch = input->dims->data[0];
  conv_params->input_height = input->dims->data[1];
  conv_params->input_width = input->dims->data[2];
  conv_params->channels_in = input->dims->data[3];

  // reading the filter dimensions
  // TODO: the following chunk of code is only true if the filter values
  // are passed in the TF format [H, W, Ci, Co] which is true only for
  // custom ops in TF lite. For builtin TF Lite ops the filters format will be
  // [Co, H, W, Ci]
  // Check input channels matching filter
  TF_LITE_ENSURE_EQ(context, conv_params->channels_in, filter->dims->data[2]);
  conv_params->filter_height = filter->dims->data[0];
  conv_params->filter_width = filter->dims->data[1];
  conv_params->channels_out = filter->dims->data[3];

  // getting the stride values. There values are already
  // stored in conv_params struct by interpreting the arguments "buffer" in
  // "Init" function
  const auto stride_height = conv_params->strides[1];
  const auto stride_width = conv_params->strides[2];

  const auto dilation_height = conv_params->dilations[1];
  const auto dilation_width = conv_params->dilations[2];

  conv_params->out_width =
      ComputeOutSize(conv_params->padding, conv_params->input_width,
                     conv_params->filter_width, stride_width, dilation_width);
  conv_params->out_height = ComputeOutSize(
      conv_params->padding, conv_params->input_height,
      conv_params->filter_height, stride_height, dilation_height);

  // determine the output dimensions
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(4);
  output_shape->data[0] = conv_params->batch;
  output_shape->data[1] = conv_params->out_height;
  output_shape->data[2] = conv_params->out_width;
  output_shape->data[3] = conv_params->channels_out;

  // allocate the output buffer
  return context->ResizeTensor(context, output, output_shape);
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, context, node);
}

template <class T, class TBitpacked>
void EvalRef(TfLiteContext* context, TfLiteNode* node,
             const TfLiteBConv2DParams* params) {
  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, T, T, TBitpacked, TBGemmFunctor>;
  using TConvFunctor =
      ce::core::Im2ColBConvFunctor<T, T, T, TFusedBGemmFunctor>;
  using PaddingFunctor = ce::core::ReferencePaddingFunctor<T, T>;

  const auto* input = GetInput(context, node, 0);
  const auto* filter = GetInput(context, node, 1);
  auto* output = GetOutput(context, node, 0);

  const auto stride_height = params->strides[1];
  const auto stride_width = params->strides[2];
  const int padding =
      params->padding == TfLitePadding::kTfLitePaddingValid ? 1 : 2;

  TConvFunctor conv_functor;
  conv_functor(input->data.f, params->batch, params->input_height,
               params->input_width, params->channels_in, filter->data.f,
               params->filter_height, params->filter_width,
               params->channels_out, stride_height, stride_width, padding,
               output->data.f, params->out_height, params->out_width);

  if (params->padding == TfLitePadding::kTfLitePaddingSame) {
      PaddingFunctor padding_functor;
      padding_functor(params->batch, params->input_height, params->input_width,
                      params->channels_in, filter->data.f,
                      params->filter_height, params->filter_width,
                      params->channels_out, stride_height, stride_width,
                      output->data.f, params->out_height, params->out_width);
  }
}

template <KernelType kernel_type, class TBitpacked>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* conv_params = reinterpret_cast<TfLiteBConv2DParams*>(node->user_data);

  if (kernel_type == KernelType::kReference) {
    EvalRef<float, TBitpacked>(context, node, conv_params);
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace bconv2d

TfLiteRegistration* Register_BCONV_2D8_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference>,
      bconv2d::Eval<bconv2d::KernelType::kReference, std::uint8_t>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D32_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference>,
      bconv2d::Eval<bconv2d::KernelType::kReference, std::uint32_t>};
  return &r;
}

TfLiteRegistration* Register_BCONV_2D64_REF() {
  static TfLiteRegistration r = {
      bconv2d::Init, bconv2d::Free,
      bconv2d::Prepare<bconv2d::KernelType::kReference>,
      bconv2d::Eval<bconv2d::KernelType::kReference, std::uint64_t>};
  return &r;
}

// use this registration wrapper to decide which impl. to use.
TfLiteRegistration* Register_BCONV_2D8() { return Register_BCONV_2D8_REF(); }
TfLiteRegistration* Register_BCONV_2D32() { return Register_BCONV_2D32_REF(); }
TfLiteRegistration* Register_BCONV_2D64() { return Register_BCONV_2D64_REF(); }

}  // namespace tflite
}  // namespace compute_engine
