// Based on the `minimal` example in tflite

#include <cstdio>
#include <cstdlib>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int get_total_elements(TfLiteIntArray* dims) {
  int total_elements = 1;
  for (int i = 0; i < dims->size; ++i) total_elements *= dims->data[i];
  return total_elements;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: test_model <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  // Check if there is exactly one input tensor and two output tensors
  TFLITE_MINIMAL_CHECK(inputs.size() == 1);
  TFLITE_MINIMAL_CHECK(outputs.size() == 2);

  int input_index = inputs[0];

  if (interpreter->tensor(input_index)->type != kTfLiteFloat32) {
    fprintf(stderr, "Can not handle this input type yet.\n");
    exit(1);
  }

  float* input_data = interpreter->typed_tensor<float>(input_index);
  TFLITE_MINIMAL_CHECK(input_data != nullptr);

  TfLiteIntArray* dims = interpreter->tensor(input_index)->dims;
  TFLITE_MINIMAL_CHECK(dims->size == 4);

  int total_elements_in = get_total_elements(dims);

  // Fill with random +1,-1 values
  for (int i = 0; i < total_elements_in; ++i)
    input_data[i] = (rand() % 2 == 0 ? 1.0f : -1.0f);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Compare the two output tensors
  auto* out0 = interpreter->tensor(outputs[0]);
  auto* out1 = interpreter->tensor(outputs[1]);
  TFLITE_MINIMAL_CHECK(out0->type == kTfLiteFloat32);
  TFLITE_MINIMAL_CHECK(out1->type == kTfLiteFloat32);

  TfLiteIntArray* dims0 = out0->dims;
  TfLiteIntArray* dims1 = out1->dims;
  TFLITE_MINIMAL_CHECK(dims0->size == dims1->size);

  int total_elements_out0 = get_total_elements(dims0);
  int total_elements_out1 = get_total_elements(dims1);
  TFLITE_MINIMAL_CHECK(total_elements_out0 == total_elements_out1);

  float* outdata0 = interpreter->typed_tensor<float>(outputs[0]);
  float* outdata1 = interpreter->typed_tensor<float>(outputs[1]);
  TFLITE_MINIMAL_CHECK(outdata0 != nullptr);
  TFLITE_MINIMAL_CHECK(outdata1 != nullptr);

  // Compare output data
  bool equal = true;
  for (int i = 0; i < total_elements_out0; ++i) {
    if (fabs(outdata0[i] - outdata1[i]) > 1e-6) {
      fprintf(stderr, "ERROR: Outputs are not equal.\n");
      equal = false;
      break;
    }
  }

  if (!equal) {
    fprintf(stderr, "Output tensor 0:\n");
    for (int i = 0; i < total_elements_out0; ++i)
      fprintf(stderr, "%f ", outdata0[i]);
    fprintf(stderr, "\nOutput tensor 1:\n");
    for (int i = 0; i < total_elements_out1; ++i)
      fprintf(stderr, "%f ", outdata1[i]);
    fprintf(stderr, "\n");
    return 1;
  }

  return 0;
}
