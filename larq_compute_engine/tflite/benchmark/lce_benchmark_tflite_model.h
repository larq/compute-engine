/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Modifications copyright (C) 2022 Larq Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef COMPUTE_ENGINE_TFLITE_BENCHMARK_LCE_BENCHMARK_TFLITE_MODEL_H_
#define COMPUTE_ENGINE_TFLITE_BENCHMARK_LCE_BENCHMARK_TFLITE_MODEL_H_

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

namespace tflite {
namespace benchmark {

// Benchmarks a TFLite model by running tflite interpreter.
class LceBenchmarkTfLiteModel : public BenchmarkTfLiteModel {
 public:
  explicit LceBenchmarkTfLiteModel(BenchmarkParams params,
                                   bool &use_reference_bconv,
                                   bool &use_indirect_bgemm);

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  static BenchmarkParams DefaultParams();
    
  using BenchmarkTfLiteModel::Run;
  TfLiteStatus Run(int argc, char** argv);
    
 private:
  bool &use_reference_bconv;
  bool &use_indirect_bgemm;

};

}  // namespace benchmark
}  // namespace tflite

#endif  // COMPUTE_ENGINE_TFLITE_BENCHMARK_LCE_BENCHMARK_TFLITE_MODEL_H_