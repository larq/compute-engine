
#include "larq_compute_engine/core/bmaxpool.h"

#include "flatbuffers/flexbuffers.h"
#include "larq_compute_engine/tflite/kernels/utils.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

using namespace tflite;

namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {
namespace bmaxpool {

using ce::core::TBitpacked;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* poolparams = new ce::ref::BMaxPoolParams{};

  const std::uint8_t* buffer_t = reinterpret_cast<const std::uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  poolparams->filter_height = m["filter_height"].AsInt32();
  poolparams->filter_width = m["filter_width"].AsInt32();
  poolparams->stride_height = m["stride_height"].AsInt32();
  poolparams->stride_width = m["stride_width"].AsInt32();
  poolparams->padding_type = ConvertPadding((Padding)m["padding"].AsInt32());

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
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, poolparams->stride_height != 0);
  TF_LITE_ENSURE(context, poolparams->stride_width != 0);
  TF_LITE_ENSURE(context, poolparams->filter_height != 0);
  TF_LITE_ENSURE(context, poolparams->filter_width != 0);

  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height;

  poolparams->padding = ComputePaddingHeightWidth(
      poolparams->stride_height, poolparams->stride_width, 1, 1, height, width,
      poolparams->filter_height, poolparams->filter_width,
      poolparams->padding_type, &out_height, &out_width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = SizeOfDimension(input, 0);
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = SizeOfDimension(input, 3);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ruy::profiler::ScopeLabel label("Binary MaxPool");

  ce::ref::BMaxPoolParams* poolparams =
      reinterpret_cast<ce::ref::BMaxPoolParams*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);

  ce::ref::BMaxPool(*poolparams, GetTensorShape(input),
                    GetTensorData<TBitpacked>(input), GetTensorShape(output),
                    GetTensorData<TBitpacked>(output));
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
