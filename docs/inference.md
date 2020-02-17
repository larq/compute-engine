# Larq Compute Engine Inference
To perform an inference with Larq Compute Engine (LCE), we use the [TensorFlow Lite
interpreter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/inference.md).
An LCE-compatible TensorFlow Lite interpreter drives the Larq model inference and
uses LCE custom operators instead of built-in TensorFlow Lite operators for each applicable
subgraph of the model.

This guide describes how to create a TensorFlow Lite interpreter with registered
LCE custom Ops and perform an inference with a [converted Larq model](./mlir_converter.md)
using LCE C++ API.

## Load and run a model in C++
Running an inference with TensorFlow Lite consists of multiple steps,
which are comprehensively described in the [TensorFlow Lite inference guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/inference.md#load-and-run-a-model-in-c).
Below we list these steps with one additional step to register LCE customs
operators using the LCE C++ function `RegisterLCECustomOps()`:

(1) Load `FlatBuffer` model:

```c++
// Load model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);
```

(2) Build the `BuiltinOpResolver` with registered LCE operators:

```c++
// create a builtin OpResolver
tflite::ops::builtin::BuiltinOpResolver resolver;

// register LCE custom ops
compute_engine::tflite::RegisterLCECustomOps(&resolver);
```

(3) Build an Interpreter with custom `OpResolver`:

```c++
// Build the interpreter
InterpreterBuilder builder(*model, resolver);
std::unique_ptr<Interpreter> interpreter;
builder(&interpreter);
```

(4) Set input tensor values:

```c++
float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

// Resize input tensors, if desired.
interpreter->AllocateTensors();
```

(5) Invoke inference:

```c++
interpreter->Invoke();
```

(6) Read inference results:

```c++
float* output = interpreter->typed_output_tensor<float>(0);
```

To build the inference binary with Bazel, it needs to be linked against `//larq_compute_engine/tflite/kernels:lce_op_kernels` target.
See [LCE minimal](../examples/lce_minimal.cc) for an example code.
