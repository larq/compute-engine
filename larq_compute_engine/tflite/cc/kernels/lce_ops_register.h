#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_LCE_REGISTER_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_LCE_REGISTER_H_

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/op_resolver.h"

// This file contains forward declaration of all custom ops
// implemented in LCE which can be used to link against LCE library.

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BSIGN();
// TfLiteRegistration* Register_BCONV_2D8();
TfLiteRegistration* Register_BCONV_2D32();
TfLiteRegistration* Register_BCONV_2D64();

// By calling this function on TF lite mutable op resolver, all LCE custom ops
// will be registerd to the op resolver.
inline void RegisterLCECustomOps(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("LqceBsign", compute_engine::tflite::Register_BSIGN());
  resolver->AddCustom("LqceBconv2d32",
                      compute_engine::tflite::Register_BCONV_2D32());
  resolver->AddCustom("LqceBconv2d64",
                      compute_engine::tflite::Register_BCONV_2D64());
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // TENSORFLOW_LITE_KERNELS_REGISTER_H_
