/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BCONV_2D8();
TfLiteRegistration* Register_BCONV_2D32();
TfLiteRegistration* Register_BCONV_2D64();

namespace testing {

namespace {

typedef TfLiteRegistration* (*register_function)(void);

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class BaseBConv2DOpModel : public SingleOpModel {
 public:
  BaseBConv2DOpModel(register_function registration, const TensorData& input,
                     const TensorData& filter, const TensorData& output,
                     int stride_width = 1, int stride_height = 1,
                     enum Padding padding = Padding_VALID,
                     int dilation_width_factor = 1,
                     int dilation_height_factor = 1, int num_threads = -1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.TypedVector("strides", [&]() {
        fbb.Int(1);
        fbb.Int(stride_height);
        fbb.Int(stride_width);
        fbb.Int(1);
      });
      fbb.TypedVector("dilations", [&]() {
        fbb.Int(1);
        fbb.Int(dilation_height_factor);
        fbb.Int(dilation_width_factor);
        fbb.Int(1);
      });
      fbb.String("filter_format", "OHWI");
      fbb.String("padding", padding == Padding_VALID ? "VALID" : "SAME");
    });
    fbb.Finish();
    SetCustomOp("LqceBconv2d", fbb.GetBuffer(), registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_)}, num_threads);
  }

 protected:
  int input_;
  int filter_;
  int output_;
};

class BConv2DOpModel : public BaseBConv2DOpModel {
 public:
  using BaseBConv2DOpModel::BaseBConv2DOpModel;

  void SetFilter(std::vector<float>& f) { PopulateTensor(filter_, f); }

  void SetInput(std::vector<float>& data) { PopulateTensor(input_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, register_function>({
    {"BConv2D8", compute_engine::tflite::Register_BCONV_2D8},
    {"BConv2D32", compute_engine::tflite::Register_BCONV_2D32},
    {"BConv2D64", compute_engine::tflite::Register_BCONV_2D64},
});

class BConv2DOpTest : public ::testing::TestWithParam<string> {
 public:
  static std::vector<string> GetKernelTags(
      const std::map<string, register_function>& kernel_map) {
    std::vector<string> tags;
    for (const auto& it : kernel_map) {
      tags.push_back(it.first);
    }
    return tags;
  }

 protected:
  const std::map<string, register_function>& GetKernelMap() {
    return *kKernelMap;
  }

  register_function GetRegistration() { return GetKernelMap().at(GetParam()); }
};

TEST_P(BConv2DOpTest, SimpleTest) {
  const int input_batch_count = 1;
  const int input_width = 4;
  const int input_height = 3;
  const int input_depth = 1;

  const int filter_height = 3;
  const int filter_width = 3;
  const int filter_count = 1;

  const int stride_width = 1;
  const int stride_height = 1;

  // TOOD: currently we are just ignoring padding
  const int pad_height = 0, pad_width = 0;
  const Padding padding = Padding_VALID;

  BConv2DOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {input_batch_count, input_height, input_width, input_depth}},
      {TensorType_FLOAT32,
       {input_depth, filter_height, filter_width, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  using T = float;
  const int input_num_elem =
      input_batch_count * input_height * input_width * input_depth;
  std::vector<T> input_data;
  input_data.resize(input_num_elem);
  std::fill(std::begin(input_data), std::end(input_data), 1);

  const int filters_num_elem =
      filter_height * filter_width * input_depth * filter_count;
  std::vector<T> filters_data;
  filters_data.resize(filters_num_elem);
  std::fill(std::begin(filters_data), std::end(filters_data), 1);

  const int output_height =
      (input_height - filter_height + 2 * pad_height) / stride_height + 1;
  const int output_width =
      (input_width - filter_width + 2 * pad_width) / stride_width + 1;
  const int output_num_elem =
      input_batch_count * output_height * output_width * filter_count;
  const int output_expected_value = input_depth * filter_height * filter_width;

  std::vector<T> output_expected;
  output_expected.resize(output_num_elem);
  std::fill(std::begin(output_expected), std::end(output_expected),
            output_expected_value);

  m.SetInput(input_data);
  m.SetFilter(filters_data);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(output_expected));
}

INSTANTIATE_TEST_SUITE_P(
    BConv2DOpTest, BConv2DOpTest,
    ::testing::ValuesIn(BConv2DOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
