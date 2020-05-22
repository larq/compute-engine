#include <fstream>
#include <iomanip>
#include <iostream>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example
// Its used in combination with the python/end2end_test.py script

using namespace tflite;

bool getFlatTensor(Interpreter* interpreter, int tensor_id, float** data,
                   int* num_elements) {
  const auto* tensor = interpreter->tensor(tensor_id);

  *data = interpreter->typed_tensor<float>(tensor_id);

  if (!*data || tensor->dims == nullptr || tensor->dims->size == 0)
    return false;

  *num_elements = 1;
  for (int i = 0; i < tensor->dims->size; ++i)
    *num_elements *= tensor->dims->data[i];

  return true;
}

#define MINIMAL_CHECK(x)                                                   \
  if (!(x)) {                                                              \
    throw std::runtime_error("Error at line " + std::to_string(__LINE__)); \
  }

std::vector<std::vector<float>> runModel(const pybind11::bytes& _flatbuffer,
                                         const std::vector<float>& input) {
  std::vector<std::vector<float>> result;

  // Load model
  const auto flatbuffer = static_cast<std::string>(_flatbuffer);
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(flatbuffer.data(),
                                               flatbuffer.size());
  MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  auto input_ids = interpreter->inputs();
  auto output_ids = interpreter->outputs();

  MINIMAL_CHECK(input_ids.size() == 1);
  MINIMAL_CHECK(output_ids.size() == 1);

  float* input_tensor_data;
  int num_input_elements;
  MINIMAL_CHECK(getFlatTensor(interpreter.get(), input_ids[0],
                              &input_tensor_data, &num_input_elements));
  MINIMAL_CHECK(num_input_elements == (int)input.size());

  float* output_tensor_data;
  int num_output_elements;
  MINIMAL_CHECK(getFlatTensor(interpreter.get(), output_ids[0],
                              &output_tensor_data, &num_output_elements));

  // Do inference twice, to check if the warmpup run and internal temporary
  // buffers work as expected
  for (int run = 0; run < 2; ++run) {
    // Fill input buffers
    for (int i = 0; i < num_input_elements; ++i) {
      input_tensor_data[i] = input[i];
    }
    // Run inference
    MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    // Store output buffers
    result.push_back(std::vector<float>(
        &output_tensor_data[0], &output_tensor_data[num_output_elements]));
  }

  return result;
}

PYBIND11_MODULE(_end2end_verify, m) { m.def("run_model", &runModel); };
