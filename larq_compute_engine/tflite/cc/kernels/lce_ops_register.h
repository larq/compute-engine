#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_LCE_REGISTER_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_LCE_REGISTER_H_

#include "tensorflow/lite/context.h"

// This file contains forward declaration of all custom ops
// implemented in LCE which can be used to link against LCE library.

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BSIGN();
// TfLiteRegistration* Register_BCONV_2D8();
TfLiteRegistration* Register_BCONV_2D32();
TfLiteRegistration* Register_BCONV_2D64();

}  // namespace tflite
}  // namespace compute_engine

#endif  // TENSORFLOW_LITE_KERNELS_REGISTER_H_
