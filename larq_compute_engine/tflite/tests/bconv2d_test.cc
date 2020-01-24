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

#include <ctime>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

// using namespace tflite;

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

class BaseConvolutionOpModel : public SingleOpModel {
 public:
  BaseConvolutionOpModel(
      TfLiteRegistration* registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, int stride_width = 2,
      int stride_height = 2, enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      int num_threads = -1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    output_ = AddOutput(output);

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      if (filter.per_channel_quantization) {
        // per channel quantization.
        std::vector<float> bias_scale(
            filter.per_channel_quantization_scales.size());
        std::vector<int64_t> bias_zero_points(
            filter.per_channel_quantization_scales.size());
        for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
             ++i) {
          bias_scale[i] =
              input.scale * filter.per_channel_quantization_scales[i];
          bias_zero_points[i] = 0;
        }
        TensorData bias{TensorType_INT32,
                        {bias_size},
                        /*min=*/0,
                        /*max=*/0,
                        /*scale=*/0,
                        /*zero_point=*/0,
                        true,
                        /*per_channel_quantization_scales=*/bias_scale,
                        /*per_channel_quantization_offsets=*/bias_zero_points,
                        /*channel_index==*/0};
        bias_ = AddInput(bias);
      } else {
        // per tensor quantization.
        auto bias_scale = GetScale(input_) * GetScale(filter_);
        TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    resolver_ = absl::make_unique<SingleOpResolver>(BuiltinOperator_CONV_2D,
                                                    registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)},
                     num_threads);
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

template <typename T, typename BiasType>
class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::vector<T>& f) { PopulateTensor(filter_, f); }

  void SetInput(std::vector<T>& data) { PopulateTensor(input_, data); }

  void SetBias(std::vector<BiasType>& f) { PopulateTensor(bias_, f); }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
};

}  // namespace
}  // namespace tflite

using namespace tflite;

namespace compute_engine {
namespace tflite {

// TfLiteRegistration* Register_BCONV_2D8();
TfLiteRegistration* Register_BCONV_2D32();
TfLiteRegistration* Register_BCONV_2D64();

namespace testing {

typedef TfLiteRegistration* (*register_function)(void);

typedef std::tuple<int,                 // batch count
                   std::array<int, 2>,  // input shape [HW]
                   int,                 // input depth
                   std::array<int, 2>,  // filter shape [HW]
                   int,                 // filter count
                   std::array<int, 2>,  // strides [HW]
                   std::array<int, 2>,  // dilations [HW]
                   Padding,             // paddding
                   int,                 // number of threads
                   std::pair<std::string, register_function>  // registration
                   >
    TestParamTuple;

struct TestParam {
  TestParam() = default;

  explicit TestParam(TestParamTuple param_tuple)
      : input_batch_count(::testing::get<0>(param_tuple)),
        input_height(::testing::get<1>(param_tuple)[0]),
        input_width(::testing::get<1>(param_tuple)[1]),
        input_depth(::testing::get<2>(param_tuple)),
        filter_height(::testing::get<3>(param_tuple)[0]),
        filter_width(::testing::get<3>(param_tuple)[1]),
        filter_count(::testing::get<4>(param_tuple)),
        stride_height(::testing::get<5>(param_tuple)[0]),
        stride_width(::testing::get<5>(param_tuple)[1]),
        dilation_height_factor(::testing::get<6>(param_tuple)[0]),
        dilation_width_factor(::testing::get<6>(param_tuple)[1]),
        padding(::testing::get<7>(param_tuple)),
        num_threads(::testing::get<8>(param_tuple)),
        kernel_name(::testing::get<9>(param_tuple).first),
        registration(::testing::get<9>(param_tuple).second) {}

  static std::string TestNameSuffix(
      const ::testing::TestParamInfo<TestParamTuple>& info) {
    const TestParam param(info.param);
    std::ostringstream param_input_oss;
    param_input_oss << param.input_batch_count << "x" << param.input_height
                    << "x" << param.input_width << "x" << param.input_depth;
    std::ostringstream param_filter_oss;
    param_filter_oss << param.filter_height << "x" << param.filter_width << "x"
                     << param.filter_count;
    std::ostringstream param_strides_oss;
    param_strides_oss << param.stride_height << "x" << param.stride_width;
    std::ostringstream param_dilation_oss;
    param_dilation_oss << param.dilation_height_factor << "x"
                       << param.dilation_width_factor;

    // WARNING: substitute accests only 11 arguments
    return absl::Substitute("Op$0_I$1_K$2_P$3_S$4_D$5_T$6", param.kernel_name,
                            param_input_oss.str(), param_filter_oss.str(),
                            param.padding == Padding_VALID ? "VALID" : "SAME",
                            param_strides_oss.str(), param_dilation_oss.str(),
                            param.num_threads);
  }

  int input_batch_count = 1;
  int input_height = 3;
  int input_width = 4;
  int input_depth = 1;

  int filter_height = 3;
  int filter_width = 3;
  int filter_count = 1;

  int stride_height = 1;
  int stride_width = 1;

  int dilation_height_factor = 1;
  int dilation_width_factor = 1;

  // TOOD: currently we are ignoring padding
  Padding padding = Padding_VALID;

  int num_threads = 1;

  std::string kernel_name = "Unknown";
  register_function registration = compute_engine::tflite::Register_BCONV_2D32;
};

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

    int channels_out = GetShape(filter_)[0];
    fused_multiply_ = AddInput({TensorType_INT32, {channels_out}});
    fused_add_ = AddInput({TensorType_INT32, {channels_out}});

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
  int fused_multiply_;
  int fused_add_;
};

template <typename T>
class BConv2DOpModel : public BaseBConv2DOpModel {
 public:
  using BaseBConv2DOpModel::BaseBConv2DOpModel;

  void SetFilter(const std::vector<T>& f) { PopulateTensor(filter_, f); }

  void SetInput(const std::vector<T>& data) { PopulateTensor(input_, data); }

  void SetFusedMultiply(std::vector<std::int32_t>& f) {
    PopulateTensor(fused_multiply_, f);
  }

  void SetFusedAdd(std::vector<std::int32_t>& f) {
    PopulateTensor(fused_add_, f);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
};

const auto kKernelMap = new std::map<string, register_function>({
    // {"BConv2D8", compute_engine::tflite::Register_BCONV_2D8},
    {"BConv2D32", compute_engine::tflite::Register_BCONV_2D32},
    {"BConv2D64", compute_engine::tflite::Register_BCONV_2D64},
});

class BConv2DOpTest : public ::testing::TestWithParam<TestParamTuple> {
 public:
  static std::vector<string> GetKernelTags(
      const std::map<string, register_function>& kernel_map) {
    std::vector<string> tags;
    for (const auto& it : kernel_map) {
      tags.push_back(it.first);
    }
    return tags;
  }

  static std::vector<register_function> GetKernelRegistrations(
      const std::map<string, register_function>& kernel_map) {
    std::vector<register_function> regs;
    for (const auto& it : kernel_map) {
      regs.push_back(it.second);
    }
    return regs;
  }

  static std::vector<std::pair<std::string, register_function>>
  GetKernelsTuples(const std::map<string, register_function>& kernel_map) {
    std::vector<std::pair<std::string, register_function>> regs;
    for (const auto& it : kernel_map) {
      regs.push_back(it);
    }
    return regs;
  }

 protected:
  const std::map<string, register_function>& GetKernelMap() {
    return *kKernelMap;
  }
};

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T, typename BiasType>
void runTest(const TestParam& param) {
  const register_function registration = param.registration;
  const int input_batch_count = param.input_batch_count;
  const int input_height = param.input_height;
  const int input_width = param.input_width;
  const int input_depth = param.input_depth;
  const int filter_height = param.filter_height;
  const int filter_width = param.filter_width;
  const int filter_count = param.filter_count;
  const int stride_height = param.stride_height;
  const int stride_width = param.stride_width;
  const int dilation_height_factor = param.dilation_height_factor;
  const int dilation_width_factor = param.dilation_width_factor;
  const Padding padding = param.padding;
  const int num_threads = param.num_threads;

  const int input_num_elem =
      input_batch_count * input_height * input_width * input_depth;

  const int filters_num_elem =
      filter_height * filter_width * input_depth * filter_count;

  // For UInt8, uint8_value can be 0 up to 255 (inclusive) and then
  //
  // real_value = scale * ( uint8_value - zero_point )
  //
  // So to get +1 and -1 values for the filter, we have
  // uint8_value = zero_point + (1 / scale)
  // uint8_value = zero_point - (1 / scale)
  //
  // We always have integers and input and as output, so we choose our scales
  // as 1.0 to simplify things.
  //
  // TODO: test more general scales
  //

  TensorType data_type = TensorType_FLOAT32;
  float input_scale = 0.0f;
  float output_scale = 0.0f;
  int32_t input_zero_point = 0;
  int32_t output_zero_point = 0;

  const bool per_channel_quantization = false;  // TODO

  if (!std::is_floating_point<T>::value) {
    data_type = TensorType_UINT8;
    input_scale = 1.0f;
    input_zero_point = 127;
    output_scale = 1.0f;
    output_zero_point = 127;
  }

  const TensorData input_tensor(
      data_type, {input_batch_count, input_height, input_width, input_depth}, 0,
      0, input_scale, input_zero_point, per_channel_quantization);

  const TensorData filter_tensor(
      data_type, {filter_count, filter_height, filter_width, input_depth}, 0, 0,
      input_scale, input_zero_point, per_channel_quantization);

  const TensorData output_tensor(data_type, {}, 0, 0, output_scale,
                                 output_zero_point, per_channel_quantization);

  std::vector<T> input_data, filters_data;
  std::vector<BiasType> bias_data;
  std::vector<std::int32_t> fused_multiply_data, fused_add_data;
  input_data.resize(input_num_elem);
  filters_data.resize(filters_num_elem);
  bias_data.resize(filter_count, 0);
  fused_multiply_data.resize(filter_count, 0);
  fused_add_data.resize(filter_count, 0);

  srand(time(NULL));
  std::array<T, 2> list{static_cast<T>(input_zero_point + 1),
                        static_cast<T>(input_zero_point - 1)};
  auto rand_generator = [&list]() {
    const int index = rand() % list.size();
    return list[index];
  };

  std::generate(std::begin(input_data), std::end(input_data), rand_generator);
  std::generate(std::begin(filters_data), std::end(filters_data),
                rand_generator);

  const std::int32_t dotproduct_size =
      filter_height * filter_width * input_depth;
  for (int i = 0; i < filter_count; ++i) {
    fused_multiply_data[i] = -2;
    fused_add_data[i] = dotproduct_size;
  }

  BConv2DOpModel<T> m_lce(registration, input_tensor, filter_tensor,
                          output_tensor, stride_width, stride_height, padding,
                          dilation_width_factor, dilation_height_factor,
                          num_threads);

  m_lce.SetInput(input_data);
  m_lce.SetFilter(filters_data);
  m_lce.SetFusedMultiply(fused_multiply_data);
  m_lce.SetFusedAdd(fused_add_data);
  m_lce.Invoke();

  ConvolutionOpModel<T, BiasType> m_builtin(
      ::tflite::ops::builtin::Register_CONVOLUTION_GENERIC_OPT(), input_tensor,
      filter_tensor, output_tensor, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor, num_threads);

  m_builtin.SetInput(input_data);
  m_builtin.SetFilter(filters_data);
  m_builtin.SetBias(bias_data);
  m_builtin.Invoke();

  EXPECT_THAT(m_lce.GetOutput(), ElementsAreArray(m_builtin.GetOutput()));
}

TEST_P(BConv2DOpTest, SimpleTest) {
  const TestParam param(GetParam());
  runTest<float, float>(param);
  // Do not yet run the int8 test if padding is SAME
  if (param.padding == Padding_VALID) {
    runTest<std::uint8_t, std::int32_t>(param);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BConv2DTests, BConv2DOpTest,
    // WARNING: ::testing::Combine accepts max 10 arguments!!!
    ::testing::Combine(
        ::testing::Values(1),  // batches
        ::testing::Values(std::array<int, 2>{7, 7},
                          std::array<int, 2>{8, 5}),  // input height/width
        ::testing::Values(1, 64, 130),                // input depth
        ::testing::Values(std::array<int, 2>{1, 1}, std::array<int, 2>{3, 3},
                          std::array<int, 2>{2, 3}),  // filter height/width
        ::testing::Values(1, 4, 64),                  // filter count
        ::testing::Values(std::array<int, 2>{1, 1},
                          std::array<int, 2>{2, 3}),  // strides height/width
        ::testing::Values(std::array<int, 2>{1, 1},
                          std::array<int, 2>{3, 2}),  // dilation height/width
        ::testing::Values(Padding_VALID, Padding_SAME),  // padding
        ::testing::Values(1, 2),                         // number of threads
        ::testing::ValuesIn(BConv2DOpTest::GetKernelsTuples(*kKernelMap))),
    TestParam::TestNameSuffix);
}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
