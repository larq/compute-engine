#ifndef COMPUTE_ENGINE_MLIR_QUANTIZE_MODEL_H_
#define COMPUTE_ENGINE_MLIR_QUANTIZE_MODEL_H_

#include <unordered_set>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"

namespace mlir {
namespace lite {

// Quantize the `input_model` and write the result to a flatbuffer `builder`.
// The `input_type` and `output_type` can be float32/qint8/int8.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const std::unordered_set<std::string>& operator_names,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter);

}  // namespace lite
}  // namespace mlir

#endif  // COMPUTE_ENGINE_MLIR_QUANTIZE_MODEL_H_
