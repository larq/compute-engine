#ifndef COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H
#define COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Converts the flatbuffer activation to what is used at runtime.
inline TfLiteFusedActivation ConvertActivation(
    ActivationFunctionType activation) {
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

// Converts the flatbuffer padding enum to TFLite padding
inline TfLitePadding ConvertPadding(Padding padding) {
  switch (padding) {
    case Padding_SAME:
      return kTfLitePaddingSame;
    case Padding_VALID:
      return kTfLitePaddingValid;
  }
  return kTfLitePaddingUnknown;
}

// Converts the TFLite padding enum to what is used at runtime.
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

}  // namespace tflite

#endif  // COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H
