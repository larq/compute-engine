#ifndef COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H
#define COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Converts the flatbuffer activation to what is used at runtime.
TfLiteFusedActivation ConvertActivation(ActivationFunctionType activation);

// Converts the flatbuffer padding enum to what is used at runtime.
TfLitePadding ConvertPadding(Padding padding);

}  // namespace tflite

#endif  // COMPUTE_ENGINE_TFLITE_KERNEL_UTILS_H
