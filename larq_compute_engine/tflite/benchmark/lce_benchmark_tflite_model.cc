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

#include "larq_compute_engine/tflite/benchmark/lce_benchmark_tflite_model.h"

#include "tensorflow/lite/tools/delegates/delegate_provider.h"


namespace tflite {
namespace benchmark {

BenchmarkParams LceBenchmarkTfLiteModel::DefaultParams() {
  BenchmarkParams default_params = BenchmarkTfLiteModel::DefaultParams();
  default_params.AddParam("use_reference_bconv",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("use_indirect_bgemm",
                          BenchmarkParam::Create<bool>(false));

  return default_params;
}

LceBenchmarkTfLiteModel::LceBenchmarkTfLiteModel(BenchmarkParams params,
                                                 bool& use_reference_bconv,
                                                 bool& use_indirect_bgemm)
    : BenchmarkTfLiteModel(std::move(params)),
      use_reference_bconv(use_reference_bconv),
      use_indirect_bgemm(use_indirect_bgemm) {}


std::vector<Flag> LceBenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkTfLiteModel::GetFlags();
  std::vector<Flag> lce_flags = {
      CreateFlag<bool>(
          "use_reference_bconv", &params_,
          "When true, uses the reference implementation of LceBconv2d."),
      CreateFlag<bool>("use_indirect_bgemm", &params_,
                       "When true, uses the optimized indirect BGEMM kernel of"
                       "LceBconv2d.")};

  flags.insert(flags.end(), lce_flags.begin(), lce_flags.end());

  return flags;
}

void LceBenchmarkTfLiteModel::LogParams() {
  BenchmarkTfLiteModel::LogParams();
  const bool verbose = params_.Get<bool>("verbose");
  LOG_BENCHMARK_PARAM(bool, "use_reference_bconv", "Use reference Bconv",
                      verbose);
  LOG_BENCHMARK_PARAM(bool, "use_indirect_bgemm", "Use indirect BGEMM",
                      verbose);
}
    
TfLiteStatus LceBenchmarkTfLiteModel::Run(int argc, char** argv) {
  TF_LITE_ENSURE_STATUS(ParseFlags(argc, argv));
  use_reference_bconv = params_.Get<bool>("use_reference_bconv");
  use_indirect_bgemm = params_.Get<bool>("use_indirect_bgemm");
    
  return BenchmarkTfLiteModel::Run();
}


}  // namespace benchmark
}  // namespace tflite