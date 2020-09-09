#ifndef COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_
#define COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/op_resolver.h"

// This file contains forward declaration of all custom ops
// implemented in LCE which can be used to link against LCE library.

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_QUANTIZE();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_BCONV_2D();
TfLiteRegistration* Register_BMAXPOOL_2D();

// By calling this function on TF lite mutable op resolver, all LCE custom ops
// will be registerd to the op resolver.
inline void RegisterLCECustomOps(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("LceQuantize",
                      compute_engine::tflite::Register_QUANTIZE());
  resolver->AddCustom("LceDequantize",
                      compute_engine::tflite::Register_DEQUANTIZE());
  resolver->AddCustom("LceBconv2d",
                      compute_engine::tflite::Register_BCONV_2D());
  resolver->AddCustom("LceBMaxPool2d",
                      compute_engine::tflite::Register_BMAXPOOL_2D());
};

}  // namespace tflite
}  // namespace compute_engine

#endif  // COMPUTE_ENGINE_TFLITE_KERNELS_LCE_OPS_REGISTER_H_
