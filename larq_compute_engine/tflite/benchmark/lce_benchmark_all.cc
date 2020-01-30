#include <benchmark/benchmark.h>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace tflite;

static void BM_model_inference(benchmark::State& state, const char* filename) {
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);

  // Allocate tensor buffers.
  interpreter->AllocateTensors();

  // warm-up
  int num_warmup_iter = 10;
  for (int i = 0; i < num_warmup_iter; ++i) interpreter->Invoke();

  // Run inference
  for (auto _ : state) {
    interpreter->Invoke();
  }
}

// NOTE: the path to the benchmarking models are hardcoded here. Modify the
// paths based on your setup.
// clang-format off
BENCHMARK_CAPTURE(BM_model_inference, mobilenet_v2_0.5_224,
                  "/data/local/tmp/mobilenet_v2_0.5_224.tflite");
BENCHMARK_CAPTURE(BM_model_inference, mobilenet_v2_0.75_224,
                  "/data/local/tmp/mobilenet_v2_0.75_224.tflite");
BENCHMARK_CAPTURE(BM_model_inference, mobilenet_v2_1.0_224,
                  "/data/local/tmp/mobilenet_v2_1.0_224.tflite");
BENCHMARK_CAPTURE(BM_model_inference, mobilenet_v2_1.0_224_quant,
                  "/data/local/tmp/mobilenet_v2_1.0_224_quant.tflite");
// clang-format on
BENCHMARK_CAPTURE(BM_model_inference, birealnet_float,
                  "/data/local/tmp/birealnet_float.tflite");
BENCHMARK_CAPTURE(BM_model_inference, birealnet_binary,
                  "/data/local/tmp//birealnet_bin.tflite");
BENCHMARK_CAPTURE(BM_model_inference, quicknet_float,
                  "/data/local/tmp/quicknet_float.tflite");
BENCHMARK_CAPTURE(BM_model_inference, quicknet_binary,
                  "/data/local/tmp/quicknet_bin.tflite");
