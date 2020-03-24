#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {

template <typename T>
T sign(T x) {
  return (x >= T(0) ? T(1) : T(-1));
}

template <>
float sign(float x) {
  return (std::signbit(x) ? -1.0f : 1.0f);
}

template <>
double sign(double x) {
  return (std::signbit(x) ? -1.0 : 1.0);
}

TfLiteStatus BsignPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus BsignEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  float* input_data = input->data.f;
  float* output_data = output->data.f;

  std::size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i = 0; i < count; ++i) {
    output_data[i] = sign(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_BSIGN() {
  static TfLiteRegistration r = {nullptr, nullptr, BsignPrepare, BsignEval};
  return &r;
}

}  // namespace tflite
}  // namespace compute_engine
