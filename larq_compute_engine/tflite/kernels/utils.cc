#include "larq_compute_engine/tflite/kernels/utils.h"

namespace tflite {

TfLiteFusedActivation ConvertActivation(ActivationFunctionType activation) {
  switch (activation) {
    case ActivationFunctionType_NONE:
      return kTfLiteActNone;
    case ActivationFunctionType_RELU:
      return kTfLiteActRelu;
    case ActivationFunctionType_RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case ActivationFunctionType_RELU6:
      return kTfLiteActRelu6;
    default:
      return kTfLiteActNone;
  }
}

TfLitePadding ConvertPadding(Padding padding) {
  switch (padding) {
    case Padding_SAME:
      return kTfLitePaddingSame;
    case Padding_VALID:
      return kTfLitePaddingValid;
  }
  return kTfLitePaddingUnknown;
}

}  // namespace tflite
