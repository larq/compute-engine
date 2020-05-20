
#include "larq_compute_engine/core/bmaxpool.h"

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/packbits_utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

using namespace tflite;

namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace bmaxpool {

using TBitpacked = std::uint32_t;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* poolparams = new ce::ref::BMaxPoolParams{};

  const std::uint8_t* buffer_t = reinterpret_cast<const std::uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  poolparams->filter_height = m["filter_height"].AsInt32();
  poolparams->filter_width = m["filter_width"].AsInt32();
  poolparams->stride_height = m["stride_h"].AsInt32();
  poolparams->stride_width = m["stride_w"].AsInt32();

  if (m["padding"].ToString() == "VALID" ||
      m["padding"].ToString() == "valid") {
    poolparams->padding_type = kTfLitePaddingValid;
  } else if (m["padding"].ToString() == "SAME" ||
             m["padding"].ToString() == "same") {
    poolparams->padding_type = kTfLitePaddingSame;
  } else {
    context->ReportError(context, "BMaxPool: invalid padding attribute.");
  }

  return poolparams;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ce::ref::BMaxPoolParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  ce::ref::BMaxPoolParams* poolparams =
      reinterpret_cast<ce::ref::BMaxPoolParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt32);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];

  int channels_out = 0;
  if (input->type == kTfLiteFloat32 || input->type == kTfLiteInt8) {
    channels_out = ce::core::GetPackedSize<TBitpacked>(input->dims->data[3]);
  } else {
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);
    channels_out = input->dims->data[3];
  }

  // Allocate temoprary tensor for bitpacked inputs
  if (input->type == kTfLiteFloat32 || input->type == kTfLiteInt8) {
    // By default, temporaries is initialized as an array of size 0
    // The logic here ensures that if we run this code the first time then
    // its properly re-allocated and when its run a second time (i.e. after a
    // tensor resize), then we keep the original temporaries array.
    if (node->temporaries && node->temporaries->size == 0) {
      TfLiteIntArrayFree(node->temporaries);
      node->temporaries = nullptr;
    }
    if (!node->temporaries) {
      node->temporaries = TfLiteIntArrayCreate(1);
      int tensor_id = -1;
      context->AddTensors(context, 1, &tensor_id);
      node->temporaries->data[0] = tensor_id;
    } else {
      TF_LITE_ENSURE_EQ(context, node->temporaries->size, 1);
    }
    TfLiteTensor* packed_input = GetTemporary(context, node, 0);

    TfLiteIntArray* packed_input_size = TfLiteIntArrayCreate(4);
    packed_input_size->data[0] = batches;
    packed_input_size->data[1] = height;
    packed_input_size->data[2] = width;
    packed_input_size->data[3] = channels_out;

    packed_input->type = kTfLiteInt32;
    packed_input->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, packed_input,
                                                     packed_input_size));
  }

  TF_LITE_ENSURE(context, poolparams->stride_height != 0);
  TF_LITE_ENSURE(context, poolparams->stride_width != 0);
  TF_LITE_ENSURE(context, poolparams->filter_height != 0);
  TF_LITE_ENSURE(context, poolparams->filter_width != 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height;

  poolparams->padding = ComputePaddingHeightWidth(
      poolparams->stride_height, poolparams->stride_width, 1, 1, height, width,
      poolparams->filter_height, poolparams->filter_width,
      poolparams->padding_type, &out_height, &out_width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ce::ref::BMaxPoolParams* poolparams =
      reinterpret_cast<ce::ref::BMaxPoolParams*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);

  const TBitpacked* packed_input_data;
  RuntimeShape packed_input_shape;

  // Note: When we update the bconv op with an optimized path for bitpacked
  // input then we might have to change the bitpack order here from Canonical to
  // Optimized
  if (input->type == kTfLiteFloat32) {
    TfLiteTensor* packed_input = GetTemporary(context, node, 0);
    ce::core::packbits_tensor<ce::core::BitpackOrder::Canonical>(
        GetTensorShape(input), GetTensorData<float>(input), 0,
        packed_input_shape, GetTensorData<TBitpacked>(packed_input));
    packed_input_data = GetTensorData<TBitpacked>(packed_input);
  } else if (input->type == kTfLiteInt8) {
    TfLiteTensor* packed_input = GetTemporary(context, node, 0);
    ce::core::packbits_tensor<ce::core::BitpackOrder::Canonical>(
        GetTensorShape(input), GetTensorData<std::int8_t>(input),
        input->params.zero_point, packed_input_shape,
        GetTensorData<TBitpacked>(packed_input));
    packed_input_data = GetTensorData<TBitpacked>(packed_input);
  } else {
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);
    packed_input_shape.ReplaceWith(4, GetTensorShape(input).DimsData());
    packed_input_data = GetTensorData<TBitpacked>(input);
  }

  ce::ref::BMaxPool(*poolparams, packed_input_shape, packed_input_data,
                    GetTensorShape(output), GetTensorData<TBitpacked>(output));
  return kTfLiteOk;
}

}  // namespace bmaxpool

TfLiteRegistration* Register_BMAXPOOL_2D() {
  static TfLiteRegistration r = {bmaxpool::Init, bmaxpool::Free,
                                 bmaxpool::Prepare, bmaxpool::Eval};
  return &r;
}

}  // namespace tflite
}  // namespace compute_engine
