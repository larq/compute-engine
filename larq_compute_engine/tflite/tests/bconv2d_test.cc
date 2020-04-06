/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
   Modifications copyright (C) 2020 Larq Contributors.

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

#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/packbits_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

// using namespace tflite;

namespace tflite {

// Use the same bitwidth as the MLIR converter
// Since tflite does not have an unsigned 32-bit int type
// we have to use the signed type here or it will throw errors.
using PackedFilterType = std::int32_t;
constexpr std::size_t packed_bitwidth = 32;

constexpr int Padding_ONE = Padding_MAX + 1;

const char* GetPaddingName(enum Padding padding) {
  switch (padding) {
    case Padding_VALID:
      return "VALID";
    case Padding_SAME:
      return "SAME";
    default:
      return "UNKNOWN";
  };
}

namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

std::string getActivationString(const enum ActivationFunctionType activation) {
  if (activation == ActivationFunctionType_RELU) {
    return "RELU";
  } else if (activation == ActivationFunctionType_NONE) {
    return "NONE";
  }
  return "UNKOWN";
}

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
    bias_ = AddInput({TensorType_FLOAT32, {bias_size}});

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

class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::vector<float>& f) { PopulateTensor(filter_, f); }

  void SetInput(std::vector<float>& data) { PopulateTensor(input_, data); }

  void SetBias(std::vector<float>& f) { PopulateTensor(bias_, f); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

// From lite/kernels/pad_test.cc
template <typename T>
class PadOpModel : public SingleOpModel {
 public:
  PadOpModel(const TensorData& input, std::initializer_list<int> paddings_shape,
             std::initializer_list<int> paddings, T constant_values,
             const TensorData& output) {
    input_ = AddInput(input);
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_shape);
    constant_values_ =
        AddConstInput(GetTensorType<T>(), {constant_values}, {1});

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                 CreatePadV2Options(builder_).Union());
    BuildInterpreter({input.shape});
  }

  void SetInput(std::vector<T>& data) { PopulateTensor<T>(input_, data); }

  void SetPaddings(std::initializer_list<int> paddings) {
    PopulateTensor<int>(paddings_, paddings);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
  int paddings_;
  int constant_values_;
};

}  // namespace
}  // namespace tflite

using namespace tflite;

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BCONV_2D32_REF();
TfLiteRegistration* Register_BCONV_2D32_OPT();
TfLiteRegistration* Register_BCONV_2D64_OPT();

namespace testing {

typedef TfLiteRegistration* (*register_function)(void);

typedef std::tuple<std::array<int, 4>,           // input shape [BHWI]
                   std::array<int, 3>,           // filter shape [HWO]
                   std::array<int, 2>,           // strides [HW]
                   std::array<int, 2>,           // dilations [HW]
                   Padding,                      // paddding
                   enum ActivationFunctionType,  // activation function
                   int,                          // number of threads
                   std::pair<std::string, register_function>>  // registration
    TestParamTuple;

struct TestParam {
  TestParam() = default;

  explicit TestParam(TestParamTuple param_tuple)
      : input_batch_count(::testing::get<0>(param_tuple)[0]),
        input_height(::testing::get<0>(param_tuple)[1]),
        input_width(::testing::get<0>(param_tuple)[2]),
        input_depth(::testing::get<0>(param_tuple)[3]),
        filter_height(::testing::get<1>(param_tuple)[0]),
        filter_width(::testing::get<1>(param_tuple)[1]),
        filter_count(::testing::get<1>(param_tuple)[2]),
        stride_height(::testing::get<2>(param_tuple)[0]),
        stride_width(::testing::get<2>(param_tuple)[1]),
        dilation_height_factor(::testing::get<3>(param_tuple)[0]),
        dilation_width_factor(::testing::get<3>(param_tuple)[1]),
        padding(::testing::get<4>(param_tuple)),
        activation(::testing::get<5>(param_tuple)),
        num_threads(::testing::get<6>(param_tuple)),
        kernel_name(::testing::get<7>(param_tuple).first),
        registration(::testing::get<7>(param_tuple).second) {}

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

    const int pad_values = (param.padding == Padding_ONE ? 1 : 0);
    const Padding padding =
        (param.padding == Padding_ONE ? Padding_SAME : param.padding);

    // WARNING: substitute accepts only 11 arguments
    return absl::Substitute(
        "Op$0_I$1_K$2_P$3_PV$4_S$5_D$6_Act$7_T$8", param.kernel_name,
        param_input_oss.str(), param_filter_oss.str(), GetPaddingName(padding),
        pad_values, param_strides_oss.str(), param_dilation_oss.str(),
        getActivationString(param.activation), param.num_threads);
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

  Padding padding = Padding_VALID;

  ::tflite::ActivationFunctionType activation = ActivationFunctionType_NONE;

  int num_threads = 1;

  std::string kernel_name = "Unknown";
  register_function registration =
      compute_engine::tflite::Register_BCONV_2D32_OPT;
};

class BaseBConv2DOpModel : public SingleOpModel {
 public:
  BaseBConv2DOpModel(
      register_function registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, int channels_in,
      int stride_width = 1, int stride_height = 1,
      enum Padding padding = Padding_VALID, int pad_values = 0,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      bool read_bitpacked_input = false, bool write_bitpacked_output = false,
      int num_threads = -1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    output_ = AddOutput(output);

    int channels_out = GetShape(filter_)[0];
    post_activation_multiplier_ =
        AddInput({TensorType_FLOAT32, {channels_out}});
    post_activation_bias_ = AddInput({TensorType_FLOAT32, {channels_out}});

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      // This attribute is necessary because if the filters are bitpacked and
      // we're reading bitpacked input then we don't have access to the original
      // 'true' number of input channels.
      fbb.Int("channels_in", channels_in);
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
      fbb.String("filter_format", "OHWI_PACKED");
      fbb.String("padding", GetPaddingName(padding));
      fbb.Int("pad_values", pad_values);
      fbb.Bool("read_bitpacked_input", read_bitpacked_input);
      fbb.Bool("write_bitpacked_output", write_bitpacked_output);
      fbb.String("activation", getActivationString(activation));
    });
    fbb.Finish();
    SetCustomOp("LceBconv2d", fbb.GetBuffer(), registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_)}, num_threads);
  }

 protected:
  int input_;
  int filter_;
  int output_;
  int post_activation_multiplier_;
  int post_activation_bias_;
};

template <typename TInput, typename TOutput>
class BConv2DOpModel : public BaseBConv2DOpModel {
 public:
  using BaseBConv2DOpModel::BaseBConv2DOpModel;

  void SetFilter(const std::vector<PackedFilterType>& f) {
    PopulateTensor(filter_, f);
  }

  void SetInput(const std::vector<TInput>& data) {
    PopulateTensor(input_, data);
  }

  void SetPostActivationMultiplier(std::vector<float>& f) {
    PopulateTensor(post_activation_multiplier_, f);
  }

  void SetPostActivationBias(std::vector<float>& f) {
    PopulateTensor(post_activation_bias_, f);
  }

  std::vector<TOutput> GetOutput() { return ExtractVector<TOutput>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

const auto kKernelMap = new std::map<string, register_function>({
    {"BConv2D32REF", compute_engine::tflite::Register_BCONV_2D32_REF},
    {"BConv2D32OPT", compute_engine::tflite::Register_BCONV_2D32_OPT},
    {"BConv2D64OPT", compute_engine::tflite::Register_BCONV_2D64_OPT},
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

// TODO: there might be a better way to define this matcher
MATCHER_P(FloatNearPointwise, tol, "Out of range") {
  return (std::get<0>(arg) > std::get<1>(arg) - tol &&
          std::get<0>(arg) < std::get<1>(arg) + tol);
}

template <typename TInput, typename TOutput>
inline void set_lce_op_input(const RuntimeShape& input_shape,
                             std::vector<float> input_data,
                             BConv2DOpModel<TInput, TOutput>& m_lce) {
  std::vector<TInput> input_data_bp;
  std::size_t input_rows_bp, input_cols_bp, input_bitpadding;
  core::packbits_matrix<core::BitpackOrder::Canonical>(
      input_data.data(), FlatSizeSkipDim(input_shape, 3), input_shape.Dims(3),
      input_data_bp, input_rows_bp, input_cols_bp, input_bitpadding,
      core::Axis::RowWise);
  m_lce.SetInput(input_data_bp);
}

// A specialisation for float input  (i.e., not reading bitpacked input).
template <typename TOutput>
inline void set_lce_op_input(const RuntimeShape& input_shape,
                             std::vector<float> input_data,
                             BConv2DOpModel<float, TOutput>& m_lce) {
  m_lce.SetInput(input_data);
}

template <typename TOutput>
inline void test_lce_op_output(const std::vector<TOutput>& lce_output_data,
                               const std::vector<int>& builtin_output_shape,
                               const std::vector<float>& builtin_output_data) {
  // Apply bitpacking to the builtin output.
  std::vector<TOutput> builtin_output_data_bp;
  std::size_t output_rows_bp, output_cols_bp, output_bitpadding;
  core::packbits_matrix<core::BitpackOrder::Canonical>(
      builtin_output_data.data(),
      builtin_output_shape.at(0) * builtin_output_shape.at(1) *
          builtin_output_shape.at(2),
      builtin_output_shape.at(3), builtin_output_data_bp, output_rows_bp,
      output_cols_bp, output_bitpadding, core::Axis::RowWise);

  // We need the outputs here to be bit-exact, so don't allow for floating
  // point imprecision.
  // It might happen that the builtin float was -0.0001 but in our op it was
  // +0.0001 in which case we get a mismatch here. If it turns out that happens,
  // we can simply add a loop here that checks the elements manually.
  EXPECT_EQ(lce_output_data, builtin_output_data_bp);
}

// A specialisation for float output (i.e., not writing bitpacked output).
inline void test_lce_op_output(const std::vector<float>& lce_output_data,
                               const std::vector<int>& builtin_output_shape,
                               const std::vector<float>& builtin_output_data) {
  EXPECT_THAT(lce_output_data, ::testing::Pointwise(FloatNearPointwise(1e-4),
                                                    builtin_output_data));
}

template <typename TInput, typename TOutput>
void runTest(const TestParam& param) {
  static_assert(std::is_same<TInput, float>::value ||
                    std::is_same<TInput, std::int8_t>::value ||
                    std::is_same<TInput, std::int32_t>::value,
                "The LCE op input type must be float or uint8 or uint32.");
  static_assert(std::is_same<TOutput, float>::value ||
                    std::is_same<TOutput, std::int8_t>::value ||
                    std::is_same<TOutput, std::int32_t>::value,
                "The LCE op output type must be float or uint8 or uint32.");

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
  const ActivationFunctionType activation = param.activation;
  const int num_threads = param.num_threads;
  const bool read_bitpacked_input = !std::is_same<TInput, float>::value;
  const bool write_bitpacked_output = !std::is_same<TOutput, float>::value;

  const Padding builtin_padding =
      (padding == Padding_ONE ? Padding_VALID : padding);
  const Padding bconv_padding =
      (padding == Padding_ONE ? Padding_SAME : padding);
  const int pad_values = (padding == Padding_ONE ? 1 : 0);

  const int input_num_elem =
      input_batch_count * input_height * input_width * input_depth;

  const int filters_num_elem =
      filter_height * filter_width * input_depth * filter_count;

  const int packed_channels =
      (input_depth + packed_bitwidth - 1) / packed_bitwidth;
  const int packed_num_elem =
      filter_count * filter_height * filter_width * packed_channels;

  // the reference implementation only support one-padding
  const auto is_reference_registration =
      (registration == compute_engine::tflite::Register_BCONV_2D32_REF);

  if ((padding == Padding_SAME && pad_values == 0) &&
      is_reference_registration) {
    GTEST_SKIP();
    return;
  }

  using T = float;
  std::vector<T> input_data, builtin_input_data, filters_data;
  std::vector<T> post_activation_multiplier_data, post_activation_bias_data,
      bias_data;
  std::vector<PackedFilterType> packed_filters_data;
  input_data.resize(input_num_elem);
  filters_data.resize(filters_num_elem);
  packed_filters_data.resize(packed_num_elem);
  bias_data.resize(filter_count, 0);
  post_activation_multiplier_data.resize(filter_count, 1);
  post_activation_bias_data.resize(filter_count, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  auto sign_generator = [&gen]() {
    return std::bernoulli_distribution(0.5)(gen) ? 1.0 : -1.0;
  };
  auto float_generator = [&gen]() {
    return std::uniform_real_distribution<>(-1.5, 1.5)(gen);
  };

  std::generate(std::begin(input_data), std::end(input_data), sign_generator);
  std::generate(std::begin(filters_data), std::end(filters_data),
                sign_generator);
  std::generate(std::begin(post_activation_multiplier_data),
                std::end(post_activation_multiplier_data), float_generator);
  std::generate(std::begin(post_activation_bias_data),
                std::end(post_activation_bias_data), float_generator);

  // Bitpack filters
  std::size_t filter_rows_bp, filter_cols_bp, filter_bitpadding;
  using namespace compute_engine::core;
  packbits_matrix<BitpackOrder::Canonical>(
      filters_data.data(), filter_count * filter_height * filter_width,
      input_depth, packed_filters_data, filter_rows_bp, filter_cols_bp,
      filter_bitpadding, Axis::RowWise);

  int padded_input_height = input_height;
  int padded_input_width = input_width;
  int output_height, output_width;
  TfLitePaddingValues padding_values = ComputePaddingHeightWidth(
      stride_height, stride_width, dilation_height_factor,
      dilation_width_factor, input_height, input_width, filter_height,
      filter_width,
      (bconv_padding == Padding_SAME ? kTfLitePaddingSame
                                     : kTfLitePaddingValid),
      &output_height, &output_width);

  if (pad_values == 1) {
    // Use a Pad op to pad with ones
    // How many pixels does the kernel stick out at the {left, top, right,
    // bottom}
    const int overflow_left = padding_values.width;
    const int overflow_top = padding_values.height;
    const int overflow_right =
        padding_values.width + padding_values.width_offset;
    const int overflow_bottom =
        padding_values.height + padding_values.height_offset;

    // The parameter {4,2} means that the paddings array is four pairs.
    // The four pairs correspond to: Batch, Height, Width, Channels
    PadOpModel<float> padop(
        {TensorType_FLOAT32,
         {input_batch_count, input_height, input_width, input_depth}},
        {4, 2},
        {0, 0, overflow_top, overflow_bottom, overflow_left, overflow_right, 0,
         0},
        1.0f, {TensorType_FLOAT32, {}});

    padop.SetInput(input_data);
    padop.Invoke();

    padded_input_height = input_height + overflow_top + overflow_bottom;
    padded_input_width = input_width + overflow_left + overflow_right;
    EXPECT_THAT(padop.GetOutputShape(),
                ElementsAreArray({input_batch_count, padded_input_height,
                                  padded_input_width, input_depth}));

    builtin_input_data = padop.GetOutput();
  } else {
    builtin_input_data = input_data;
  }

  if (padding == Padding_SAME &&
      (activation == ActivationFunctionType_RELU || write_bitpacked_output)) {
    // Neither fused ReLu nor writing bitpacked output is supported with
    // zero-padding. We could use `EXPECT_DEATH` here but it is extremely slow.
    // Therefore we have a separate test below, and here we just return.
    return;
  }

  /*-----------------
    Run built-in op.
   -----------------*/

  ConvolutionOpModel m_builtin(
      ::tflite::ops::builtin::
          Register_CONVOLUTION_GENERIC_OPT(),  // registration
      {TensorType_FLOAT32,
       {input_batch_count, padded_input_height, padded_input_width,
        input_depth}},  // input
      {TensorType_FLOAT32,
       {filter_count, filter_height, filter_width, input_depth}},  // filter
      {TensorType_FLOAT32, {}},                                    // output
      stride_width, stride_height, builtin_padding, activation,
      dilation_width_factor, dilation_height_factor, num_threads);

  m_builtin.SetInput(builtin_input_data);
  m_builtin.SetFilter(filters_data);
  m_builtin.SetBias(bias_data);
  m_builtin.Invoke();
  auto builtin_output = m_builtin.GetOutput();

  // Apply the post multiply and add to the TFLite model.
  // We cannot fuse it into the tflite bias because it should happen *after*
  // the activation function.
  T* out_ptr = builtin_output.data();
  for (int batch = 0; batch < input_batch_count; ++batch) {
    for (int out_y = 0; out_y < output_width; ++out_y) {
      for (int out_x = 0; out_x < output_height; ++out_x) {
        for (int out_c = 0; out_c < filter_count; ++out_c) {
          *out_ptr *= post_activation_multiplier_data[out_c];
          *out_ptr += post_activation_bias_data[out_c];
          ++out_ptr;
        }
      }
    }
  }

  /*-------------
    Test LCE op.
   -------------*/

  // Create LCE op.

  TensorType input_tensor_type;
  if (std::is_same<TInput, float>::value)
    input_tensor_type = TensorType_FLOAT32;
  else if (std::is_same<TInput, std::int8_t>::value)
    input_tensor_type = TensorType_INT8;
  else
    input_tensor_type = TensorType_INT32;

  TensorType output_tensor_type;
  if (std::is_same<TOutput, float>::value)
    output_tensor_type = TensorType_FLOAT32;
  else if (std::is_same<TOutput, std::int8_t>::value)
    output_tensor_type = TensorType_INT8;
  else
    output_tensor_type = TensorType_INT32;

  // input_depth is always unpacked
  // For reading bitpacked input, the input tensor can be packed
  int input_tensor_channels = input_depth;
  if (read_bitpacked_input) {
    // We can't use `std::numeric_limits<TInput>` here because we bitpack into
    // *signed* integers, and the bitwidth will be one less than expected.
    const auto bitwidth = sizeof(TInput) * CHAR_BIT;
    input_tensor_channels = (input_depth + bitwidth - 1) / bitwidth;
  }

  BConv2DOpModel<TInput, TOutput> m_lce(
      registration,
      {input_tensor_type,
       {input_batch_count, input_height, input_width, input_tensor_channels}},
      {TensorType_INT32,
       {filter_count, filter_height, filter_width, packed_channels}},
      {output_tensor_type, {}}, input_depth, stride_width, stride_height,
      bconv_padding, pad_values, activation, dilation_width_factor,
      dilation_height_factor, read_bitpacked_input, write_bitpacked_output,
      num_threads);

  // Set op parameters.

  set_lce_op_input({input_batch_count, input_height, input_width, input_depth},
                   input_data, m_lce);
  m_lce.SetFilter(packed_filters_data);
  m_lce.SetPostActivationMultiplier(post_activation_multiplier_data);
  m_lce.SetPostActivationBias(post_activation_bias_data);

  // Invoke the op and test that the output is correct.

  m_lce.Invoke();
  test_lce_op_output(m_lce.GetOutput(), m_builtin.GetOutputShape(),
                     builtin_output);
}

TEST_P(BConv2DOpTest, ReadFPWriteFP) {
  runTest<float, float>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadBPWriteBP) {
  runTest<std::int32_t, std::int32_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadFPWriteBP) {
  runTest<float, std::int32_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadBPWriteFP) {
  runTest<std::int32_t, float>(TestParam(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    BConv2DTests, BConv2DOpTest,
    // WARNING: ::testing::Combine accepts max 10 arguments!!!
    ::testing::Combine(
        ::testing::Values(
            std::array<int, 4>{1, 7, 7, 1}, std::array<int, 4>{1, 8, 5, 1},
            std::array<int, 4>{1, 7, 7, 64}, std::array<int, 4>{1, 8, 5, 64},
            std::array<int, 4>{1, 7, 7, 130},
            std::array<int, 4>{1, 8, 5, 130}),  // input shape [BHWI]
        ::testing::Values(std::array<int, 3>{1, 1, 1},
                          std::array<int, 3>{3, 3, 1},
                          std::array<int, 3>{2, 3, 1},
                          std::array<int, 3>{1, 1, 4},
                          std::array<int, 3>{3, 3, 4},
                          std::array<int, 3>{2, 3, 4},
                          std::array<int, 3>{1, 1, 64},
                          std::array<int, 3>{3, 3, 64},
                          std::array<int, 3>{2, 3, 64}),  // filter shape [HWO]
        ::testing::Values(std::array<int, 2>{1, 1},
                          std::array<int, 2>{2, 3}),  // strides height/width
        ::testing::Values(std::array<int, 2>{1, 1},
                          std::array<int, 2>{3, 2}),  // dilation height/width
        ::testing::Values(Padding_VALID, Padding_SAME, Padding_ONE),  // padding
        ::testing::Values(ActivationFunctionType_NONE,
                          ActivationFunctionType_RELU),  // activation function
        ::testing::Values(1, 2),                         // number of threads
        ::testing::ValuesIn(BConv2DOpTest::GetKernelsTuples(*kKernelMap))),
    TestParam::TestNameSuffix);

TEST(BConv2DTests, BConvErrorTest) {
  // We have to use typedefs or else the template invocation in the type
  // confuses the pre-processor (EXPECT_DEATH is a macro).
  typedef BConv2DOpModel<float, float> FP_BConv2DOpModel;
  typedef BConv2DOpModel<float, std::int8_t> Bitpacked_BConv2DOpModel;

  // Test if fused ReLu throws an error in combination with zero-padding.
  EXPECT_DEATH(
      {
        FP_BConv2DOpModel m_lce(compute_engine::tflite::Register_BCONV_2D64_OPT,
                                {TensorType_FLOAT32, {1, 16, 16, 64}},
                                {TensorType_FLOAT32, {128, 3, 3, 64}},
                                {TensorType_FLOAT32, {}}, 64, 1, 1,
                                Padding_SAME, 0, ActivationFunctionType_RELU, 1,
                                1, false, false, 1);
      },
      "Fused activations are only supported with valid or one-padding.");

  // Test if writing bitpacked output throws an error in combination with
  // zero-padding.
  EXPECT_DEATH(
      {
        Bitpacked_BConv2DOpModel m_lce(
            compute_engine::tflite::Register_BCONV_2D64_OPT,
            {TensorType_FLOAT32, {1, 16, 16, 64}},
            {TensorType_FLOAT32, {128, 3, 3, 64}}, {TensorType_UINT8, {}}, 64,
            1, 1, Padding_SAME, 0, ActivationFunctionType_NONE, 1, 1, false,
            true, 1);
      },
      "Writing bitpacked output is only supported with valid or one-padding.");
}

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
