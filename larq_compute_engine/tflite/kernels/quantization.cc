#include "larq_compute_engine/core/bitpack_utils.h"
#include "ruy/profiler/instrumentation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

using namespace tflite;

namespace ce = compute_engine;

namespace compute_engine {
namespace tflite {

using ce::core::TBitpacked;

TfLiteStatus QuantizePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt32);

  int num_dims = NumDimensions(input);
  TF_LITE_ENSURE_EQ(context, num_dims, NumDimensions(output));

  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);

  // The last dimension is bitpacked
  output_dims->data[num_dims - 1] =
      ce::core::GetBitpackedSize(SizeOfDimension(input, num_dims - 1));

  return context->ResizeTensor(context, output, output_dims);
}

TfLiteStatus DequantizePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt32);
  TF_LITE_ENSURE(context,
                 output->type == kTfLiteFloat32 || output->type == kTfLiteInt8);

  int num_dims = NumDimensions(input);

  TF_LITE_ENSURE_EQ(context, num_dims, NumDimensions(output));

  // The first n-1 dimensions are equal
  for (int i = 0; i < num_dims - 1; ++i) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(output, i),
                      SizeOfDimension(input, i));
  }
  // The last dimension is bitpacked
  int packed_channels = SizeOfDimension(input, num_dims - 1);
  int unpacked_channels = SizeOfDimension(output, num_dims - 1);
  TF_LITE_ENSURE_EQ(context, packed_channels,
                    ce::core::GetBitpackedSize(unpacked_channels));

  // We don't support resizing here, because we can not know the number of
  // output channels based on the number of input channels

  return kTfLiteOk;
}

TfLiteStatus QuantizeEval(TfLiteContext* context, TfLiteNode* node) {
  ruy::profiler::ScopeLabel label("Binary Quantize");

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  if (input->type == kTfLiteFloat32) {
    ce::core::bitpack_tensor(GetTensorShape(input), GetTensorData<float>(input),
                             0, GetTensorData<TBitpacked>(output));
  } else if (input->type == kTfLiteInt8) {
    ce::core::bitpack_tensor(
        GetTensorShape(input), GetTensorData<std::int8_t>(input),
        input->params.zero_point, GetTensorData<TBitpacked>(output));
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus DequantizeEval(TfLiteContext* context, TfLiteNode* node) {
  ruy::profiler::ScopeLabel label("Binary Dequantize");

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto out_shape = GetTensorShape(output);
  int dims = out_shape.DimensionsCount();
  int num_rows = FlatSizeSkipDim(out_shape, dims - 1);
  int num_cols = out_shape.Dims(dims - 1);

  if (output->type == kTfLiteFloat32) {
    ce::core::unpack_matrix(GetTensorData<TBitpacked>(input), num_rows,
                            num_cols, GetTensorData<float>(output));
  } else if (output->type == kTfLiteInt8) {
    int offset = TfLiteRound(1.0f / output->params.scale);
    std::int8_t zero_bit_result =
        std::min(127, output->params.zero_point + offset);
    std::int8_t one_bit_result =
        std::max(-128, output->params.zero_point - offset);
    ce::core::unpack_matrix(GetTensorData<TBitpacked>(input), num_rows,
                            num_cols, GetTensorData<std::int8_t>(output),
                            zero_bit_result, one_bit_result);
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteRegistration* Register_QUANTIZE() {
  static TfLiteRegistration r = {nullptr, nullptr, QuantizePrepare,
                                 QuantizeEval};
  return &r;
}

TfLiteRegistration* Register_DEQUANTIZE() {
  static TfLiteRegistration r = {nullptr, nullptr, DequantizePrepare,
                                 DequantizeEval};
  return &r;
}

}  // namespace tflite
}  // namespace compute_engine
