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
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/core/packbits_utils.h"
#include "larq_compute_engine/tflite/tests/bconv2d_op_model.h"
#include "larq_compute_engine/tflite/tests/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/padding.h"
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
      // of input and filter. This is correctly set during conversion.
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
                     num_threads, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

template <typename T>
class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::vector<T>& f) { PopulateTensor(filter_, f); }

  void SetInput(std::vector<T>& data) { PopulateTensor(input_, data); }

  void SetBias(std::vector<typename GetBiasType<T>::type>& f) {
    PopulateTensor(bias_, f);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
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
    // The constant_values_ input is required to have the same int8 quantization
    // parameters (scale, zero_point) as the output tensor
    TensorData cv_tensor = output;
    cv_tensor.shape = {1};
    constant_values_ = AddConstInput(cv_tensor, {constant_values});

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

TfLiteRegistration* Register_BCONV_2D_32_REF();
TfLiteRegistration* Register_BCONV_2D_32_OPT();
TfLiteRegistration* Register_BCONV_2D_64_OPT();

namespace testing {

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
      compute_engine::tflite::Register_BCONV_2D_32_OPT;
};

const auto kKernelMap = new std::map<string, register_function>({
    {"BConv2D32REF", compute_engine::tflite::Register_BCONV_2D_32_REF},
    {"BConv2D32OPT", compute_engine::tflite::Register_BCONV_2D_32_OPT},
    {"BConv2D64OPT", compute_engine::tflite::Register_BCONV_2D_64_OPT},
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

void ComputeThresholds(int input_depth, int filter_height, int filter_width,
                       const std::vector<float>& post_activation_multiplier,
                       const std::vector<float>& post_activation_bias,
                       enum ActivationFunctionType activation,
                       std::vector<std::int32_t>& threshold) {
  // Precompute thresholds
  // fp_result = post_bias + post_mul * clamp(backtransform - 2 * accumulator)
  // The output bit is 1 iff fp_result < 0
  // So when -post_bias/post_mul is within the clamp range, then this means:
  // The output bit is 1 iff:
  // accumulator > (backtransform_add + post_bias / post_mul) / 2.0f

  std::int32_t output_activation_min, output_activation_max;
  if (activation == ActivationFunctionType_RELU) {
    output_activation_min = 0;
    output_activation_max = std::numeric_limits<std::int32_t>::max();
  } else {
    output_activation_min = std::numeric_limits<std::int32_t>::lowest();
    output_activation_max = std::numeric_limits<std::int32_t>::max();
  }

  // We do all intermediate computations here in double to keep accuracy
  const double backtransform_add = filter_height * filter_width * input_depth;
  for (size_t i = 0; i < post_activation_multiplier.size(); ++i) {
    const double post_mul = post_activation_multiplier[i];
    const double post_bias = post_activation_bias[i];

    const double t1 = -post_bias / post_mul;
    const double t2 = (0.5 * (backtransform_add + post_bias / post_mul));
    // The rounding direction is important here.
    // We have to use a truncating cast instead of a round
    threshold[i] = static_cast<std::int32_t>(t2);

    if (t2 >= 2 * backtransform_add || t1 <= output_activation_min) {
      threshold[i] = std::numeric_limits<std::int32_t>::max();
    } else if (t2 <= 0.0 || t1 >= output_activation_max) {
      threshold[i] = std::numeric_limits<std::int32_t>::lowest();
    }
  }
}

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// TODO: there might be a better way to define this matcher
MATCHER_P(FloatNearPointwise, tol, "Out of range") {
  return (std::get<0>(arg) > std::get<1>(arg) - tol &&
          std::get<0>(arg) < std::get<1>(arg) + tol);
}

// 8-bit quantized or float input
template <typename TInput, typename PostType, typename TOutput>
void set_lce_op_input(
    const RuntimeShape& input_shape,
    std::vector<typename GetBuiltinType<TInput, TOutput>::type> input_data,
    std::int32_t zero_point, BConv2DOpModel<TInput, PostType, TOutput>& m_lce) {
  m_lce.SetInput(input_data);
}

// int32 input
template <typename TData, typename PostType, typename TOutput>
void set_lce_op_input(const RuntimeShape& input_shape,
                      std::vector<TData> input_data, std::int32_t zero_point,
                      BConv2DOpModel<std::int32_t, PostType, TOutput>& m_lce) {
  std::vector<std::int32_t> input_data_bp(
      core::GetPackedTensorSize<std::int32_t>(input_shape));
  RuntimeShape output_shape;
  core::packbits_tensor(input_shape, input_data.data(), zero_point,
                        output_shape, input_data_bp.data());
  m_lce.SetInput(input_data_bp);
}

// Output test for writing bitpacked output
template <typename BuiltinType>
void test_lce_op_output(const std::vector<std::int32_t>& lce_output_data,
                        const std::vector<int>& builtin_output_shape,
                        const std::vector<BuiltinType>& builtin_output_data,
                        std::int32_t zero_point) {
  // Apply bitpacking to the builtin output.
  RuntimeShape out_shape;
  out_shape.BuildFrom(builtin_output_shape);
  std::vector<std::int32_t> builtin_output_data_bp(
      core::GetPackedTensorSize<std::int32_t>(out_shape));
  RuntimeShape packed_shape;
  core::packbits_tensor(out_shape, builtin_output_data.data(), zero_point,
                        packed_shape, builtin_output_data_bp.data());

  // We need the outputs here to be bit-exact, so don't allow for floating
  // point imprecision.
  // It might happen that the builtin float was -0.0001 but in our op it was
  // +0.0001 in which case we get a mismatch here. If it turns out that happens,
  // we can simply add a loop here that checks the elements manually.
  EXPECT_EQ(lce_output_data, builtin_output_data_bp);
}

// Output test for 8-bit quantized or float output
template <typename TOutput>
void test_lce_op_output(const std::vector<TOutput>& lce_output_data,
                        const std::vector<int>& builtin_output_shape,
                        const std::vector<TOutput>& builtin_output_data,
                        std::int32_t zero_point) {
  EXPECT_THAT(lce_output_data, ::testing::Pointwise(FloatNearPointwise(1e-3),
                                                    builtin_output_data));
}

template <typename TInput, typename TOutput>
void runTest(const TestParam& param) {
  static_assert(std::is_same<TInput, float>::value ||
                    std::is_same<TInput, std::int8_t>::value ||
                    std::is_same<TInput, std::int32_t>::value,
                "The LCE op input type must be float or int8 or int32.");
  static_assert(std::is_same<TOutput, float>::value ||
                    std::is_same<TOutput, std::int8_t>::value ||
                    std::is_same<TOutput, std::int32_t>::value,
                "The LCE op output type must be float or int8 or int32.");

  using BuiltinType = typename GetBuiltinType<TInput, TOutput>::type;
  using BuiltinBiasType = typename GetBiasType<BuiltinType>::type;
  using PostType = typename GetPostType<BuiltinType>::type;

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
  constexpr bool read_bitpacked_input =
      std::is_same<TInput, std::int32_t>::value;
  constexpr bool write_bitpacked_output =
      std::is_same<TOutput, std::int32_t>::value;
  constexpr bool quantized_model = (std::is_same<TInput, std::int8_t>::value ||
                                    std::is_same<TOutput, std::int8_t>::value);

  // 8-bit quantization if and only if the builtin type is int8
  static_assert(quantized_model ==
                std::is_same<BuiltinType, std::int8_t>::value);

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
      core::GetPackedSize<PackedFilterType>(input_depth);
  const int packed_num_elem =
      filter_count * filter_height * filter_width * packed_channels;

  // the reference implementation only support one-padding
  const auto is_reference_registration =
      (registration == compute_engine::tflite::Register_BCONV_2D_32_REF);

  if ((padding == Padding_SAME && pad_values == 0) &&
      is_reference_registration) {
    GTEST_SKIP();
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  LceTensor<BuiltinType> input_tensor(
      {input_batch_count, input_height, input_width, input_depth});

  // Shape will be changed later if padding is required
  LceTensor<BuiltinType> padded_input_tensor(
      {input_batch_count, input_height, input_width, input_depth});

  LceTensor<BuiltinType> filter_tensor(
      {filter_count, filter_height, filter_width, input_depth});

  LceTensor<std::int32_t> packed_filter_tensor(
      {filter_count, filter_height, filter_width, packed_channels});

  // We can use the same tensor object for multiply and bias
  // because they have the same shape and datatype
  LceTensor<PostType> post_tensor({filter_count});

  LceTensor<std::int32_t> threshold_tensor({filter_count});

  LceTensor<TOutput> output_tensor;
  LceTensor<BuiltinType> builtin_output_tensor;

  if (quantized_model) {
    // Note: it can still be that we have bitpacked input or output

    input_tensor.GenerateQuantizationParams(gen);
    // Use the same quantization parameters for the builtin op
    padded_input_tensor.SetQuantizationParams(input_tensor);
    filter_tensor.GenerateQuantizationParamsPerChannel(gen);
    builtin_output_tensor.GenerateQuantizationParams(gen);

    if (!write_bitpacked_output)
      output_tensor.SetQuantizationParams(builtin_output_tensor);

    if (std::is_same<PostType, std::int8_t>::value) {
      post_tensor.GenerateQuantizationParams(gen);
    }

    // In TF 2.1, Relu is broken on int8 Conv2D.
    // They fixed it in this commit, which is in TF 2.2
    // https://github.com/tensorflow/tensorflow/commit/25adce3551d145f615f77eafd08159451e5be0c8
    // Until this branch is rebased on TF 2.2,
    // we skip the Relu int8 test
    if (activation == ActivationFunctionType_RELU) {
      GTEST_SKIP();
      return;
    }
  }

  std::vector<BuiltinType> input_data, padded_input_data, filters_data;
  std::vector<PostType> post_activation_multiplier_data,
      post_activation_bias_data;
  std::vector<std::int32_t> threshold_data;
  std::vector<BuiltinBiasType> bias_data;

  std::vector<PackedFilterType> packed_filters_data;

  input_data.resize(input_num_elem);
  filters_data.resize(filters_num_elem);
  packed_filters_data.resize(packed_num_elem);
  bias_data.resize(
      filter_count,
      0);  // bias always has zero_point = 0, scale = input * filter
  post_activation_multiplier_data.resize(filter_count, 1);
  post_activation_bias_data.resize(filter_count, 0);
  threshold_data.resize(filter_count);

  // Fill input and filter with -1,+1 with quantization taken into account
  input_tensor.GenerateSigns(gen, std::begin(input_data), std::end(input_data));
  filter_tensor.GenerateSigns(gen, std::begin(filters_data),
                              std::end(filters_data));

  if (quantized_model) {
    // Set the post_activation_ step to identity, to make sure
    // the Conv2D and BConv2D kernel are clamping at the same point.
    // We have a separate test for non-identity post_activation_
    for (int i = 0; i < filter_count; ++i) {
      post_activation_multiplier_data[i] = post_tensor.Quantize(1);
      post_activation_bias_data[i] = post_tensor.Quantize(0);
    }
  } else {  // Bitpacked input or float input
    auto float_generator = [&gen]() {
      return std::uniform_real_distribution<>(0.01, 1.5)(gen);
    };

    std::generate(std::begin(post_activation_multiplier_data),
                  std::end(post_activation_multiplier_data), float_generator);
    std::generate(std::begin(post_activation_bias_data),
                  std::end(post_activation_bias_data), float_generator);
  }

  if (write_bitpacked_output) {
    ComputeThresholds(input_depth, filter_height, filter_width,
                      post_activation_multiplier_data,
                      post_activation_bias_data, activation, threshold_data);
  }

  // Bitpack filters
  using namespace compute_engine::core;
  packbits_matrix(filters_data.data(),
                  filter_count * filter_height * filter_width, input_depth,
                  packed_filters_data.data());

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

    int padded_input_height = input_height + overflow_top + overflow_bottom;
    int padded_input_width = input_width + overflow_left + overflow_right;

    // The parameter {4,2} means that the paddings array is four pairs.
    // The four pairs correspond to: Batch, Height, Width, Channels
    const BuiltinType pad_value = input_tensor.Quantize(1);
    PadOpModel<BuiltinType> padop(input_tensor, {4, 2},
                                  {0, 0, overflow_top, overflow_bottom,
                                   overflow_left, overflow_right, 0, 0},
                                  pad_value, padded_input_tensor);

    padop.SetInput(input_data);
    padop.Invoke();

    EXPECT_THAT(padop.GetOutputShape(),
                ElementsAreArray({input_batch_count, padded_input_height,
                                  padded_input_width, input_depth}));

    padded_input_data = padop.GetOutput();
    padded_input_tensor.shape = {input_batch_count, padded_input_height,
                                 padded_input_width, input_depth};
  } else {
    padded_input_data = input_data;
  }

  if (padding == Padding_SAME &&
      (activation == ActivationFunctionType_RELU || write_bitpacked_output ||
       std::is_same<BuiltinType, std::int8_t>::value)) {
    // Neither fused ReLu nor writing bitpacked output nor int8 is supported
    // with zero-padding. We could use `EXPECT_DEATH` here but it is extremely
    // slow. Therefore we have a separate test below, and here we just skip.
    GTEST_SKIP();
    return;
  }

  /*-----------------
    Run built-in op.
   -----------------*/

  ConvolutionOpModel<BuiltinType> m_builtin(
      ::tflite::ops::builtin::Register_CONVOLUTION_GENERIC_OPT(),
      padded_input_tensor, filter_tensor, builtin_output_tensor, stride_width,
      stride_height, builtin_padding, activation, dilation_width_factor,
      dilation_height_factor, num_threads);

  m_builtin.SetInput(padded_input_data);
  m_builtin.SetFilter(filters_data);
  m_builtin.SetBias(bias_data);
  m_builtin.Invoke();
  auto builtin_output = m_builtin.GetOutput();

  // Apply the post multiply and add to the TFLite model.
  // We cannot fuse it into the tflite bias because it should happen *after*
  // the activation function.
  //
  // The bitpacked-output case does not require such processing.
  // The 8-bit quantized case:
  //
  // BConv: clamp(Scale&Round(post_add) + Scale&Round(post_mul * accumulator))
  // Conv:  clamp(Scale&Round( accumulator ))
  //
  // There are several ways in which we can have a *legitimate* mismatch
  // between the BConv and Conv outputs:
  // - conv output got clamped
  // - bconv output got clamped
  // - scale&round rounding effect of conv
  // - scale&round rounding effect of post_add or post_mul etc
  //
  // Therefore we simply set the post_mul and post_add to identity in this
  // test, and we have a separate test below to test a non-identity version
  if (std::is_same<BuiltinType, float>::value) {
    BuiltinType* out_ptr = builtin_output.data();
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
  }

  /*-------------
    Test LCE op.
   -------------*/

  // Create LCE op.

  // For reading bitpacked input, the input tensor is bitpacked
  if (read_bitpacked_input) {
    // We can't use `std::numeric_limits<TInput>` here because we bitpack into
    // *signed* integers, and the bitwidth will be one less than expected.
    const auto bitwidth = sizeof(TInput) * CHAR_BIT;
    input_tensor.shape[3] = (input_depth + bitwidth - 1) / bitwidth;
    input_tensor.type = TensorType_INT32;
  }

  BConv2DOpModel<TInput, PostType, TOutput> m_lce(
      registration, input_tensor, packed_filter_tensor, output_tensor,
      post_tensor, post_tensor, threshold_tensor, input_depth, stride_width,
      stride_height, bconv_padding, pad_values, activation,
      dilation_width_factor, dilation_height_factor, num_threads);

  // Set op parameters.
  set_lce_op_input({input_batch_count, input_height, input_width, input_depth},
                   input_data, input_tensor.zero_point, m_lce);
  m_lce.SetFilter(packed_filters_data);
  if (write_bitpacked_output) {
    m_lce.SetThresholds(threshold_data);
  } else {
    m_lce.SetPostActivationMultiplier(post_activation_multiplier_data);
    m_lce.SetPostActivationBias(post_activation_bias_data);
  }

  // Invoke the op and test that the output is correct.
  m_lce.Invoke();
  test_lce_op_output(m_lce.GetOutput(), m_builtin.GetOutputShape(),
                     builtin_output, builtin_output_tensor.zero_point);
}

// Three input types: int8, int32, float
// Three output types: int8, int32, float
// Thats 9 combinations, but float -> int8 and int8 -> float are never used so
// that leaves 7 combinations.

TEST_P(BConv2DOpTest, ReadFloatWriteFloat) {
  runTest<float, float>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadBPWriteBP) {
  runTest<std::int32_t, std::int32_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadFloatWriteBP) {
  runTest<float, std::int32_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadBPWriteFloat) {
  runTest<std::int32_t, float>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadInt8WriteInt8) {
  runTest<std::int8_t, std::int8_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadBPWriteInt8) {
  runTest<std::int32_t, std::int8_t>(TestParam(GetParam()));
}

TEST_P(BConv2DOpTest, ReadInt8WriteBP) {
  runTest<std::int8_t, std::int32_t>(TestParam(GetParam()));
}

using ::testing::Values;
using ::testing::ValuesIn;

// For testing it's convenient to first run a small test on all the different
// types.
INSTANTIATE_TEST_SUITE_P(
    SmallTest, BConv2DOpTest,
    ::testing::Combine(
        Values(std::array<int, 4>{1, 4, 4, 1},
               std::array<int, 4>{1, 4, 4, 64}),  // input shape [BHWI]
        Values(std::array<int, 3>{1, 1, 1}, std::array<int, 3>{3, 3, 1},
               std::array<int, 3>{3, 3, 64}),  // filter shape [HWO]
        Values(std::array<int, 2>{1, 1}),      // strides height/width
        Values(std::array<int, 2>{1, 1}),      // dilation height/width
        Values(Padding_VALID, Padding_ONE),    // padding
        Values(ActivationFunctionType_NONE),   // activation function
        Values(1),                             // number of threads
        ValuesIn(BConv2DOpTest::GetKernelsTuples(*kKernelMap))),
    TestParam::TestNameSuffix);

#if RUY_PLATFORM_ARM_64
// Separately, for 64-bit optimised kernels only, test a very large input
// channel and filter size combination that would overflow 16-bit accumulators
// (to check that we successfully fall back to the 32-bit accumulator kernels).
INSTANTIATE_TEST_SUITE_P(
    OptimizedKernel16BitOverflowTest, BConv2DOpTest,
    ::testing::Combine(
        Values(std::array<int, 4>{1, 8, 8, 2048}),  // input shape [BHWI]
        Values(std::array<int, 3>{7, 7, 4}),        // filter shape [HWO]
        Values(std::array<int, 2>{1, 1}),           // strides height/width
        Values(std::array<int, 2>{1, 1}),           // dilation height/width
        Values(Padding_VALID, Padding_ONE),         // padding
        Values(ActivationFunctionType_NONE,
               ActivationFunctionType_RELU),  // activation function
        Values(1, 2),                         // number of threads
        Values(std::pair<std::string, register_function>{
            "BConv2D64OPT", compute_engine::tflite::Register_BCONV_2D_64_OPT})),
    TestParam::TestNameSuffix);
#endif

// The BigTest suite will be skipped in the qemu CI runs as they take more than
// an hour.
INSTANTIATE_TEST_SUITE_P(
    BigTest, BConv2DOpTest,
    ::testing::Combine(
        Values(std::array<int, 4>{1, 7, 7, 1}, std::array<int, 4>{1, 8, 5, 1},
               std::array<int, 4>{1, 7, 7, 64}, std::array<int, 4>{1, 8, 5, 64},
               std::array<int, 4>{1, 7, 7, 130},
               std::array<int, 4>{1, 8, 5, 130}),  // input shape [BHWI]
        Values(std::array<int, 3>{1, 1, 1}, std::array<int, 3>{3, 3, 1},
               std::array<int, 3>{2, 3, 1}, std::array<int, 3>{1, 1, 4},
               std::array<int, 3>{3, 3, 4}, std::array<int, 3>{2, 3, 4},
               std::array<int, 3>{1, 1, 64}, std::array<int, 3>{3, 3, 64},
               std::array<int, 3>{2, 3, 64}),  // filter shape [HWO]
        Values(std::array<int, 2>{1, 1},
               std::array<int, 2>{2, 3}),  // strides height/width
        Values(std::array<int, 2>{1, 1},
               std::array<int, 2>{3, 2}),  // dilation height/width
        Values(Padding_VALID, Padding_SAME, Padding_ONE),  // padding
        Values(ActivationFunctionType_NONE,
               ActivationFunctionType_RELU),  // activation function
        Values(1, 2),                         // number of threads
        ValuesIn(BConv2DOpTest::GetKernelsTuples(*kKernelMap))),
    TestParam::TestNameSuffix);

TEST(BConv2DTests, ReluErrorTest) {
  LceTensor<float> input_tensor({1, 16, 16, 64});
  LceTensor<std::int32_t> packed_filter_tensor({128, 3, 3, 64});
  LceTensor<float> post_tensor({128});
  LceTensor<std::int32_t> threshold_tensor({128});
  LceTensor<float> output_tensor;
  LceTensor<std::int32_t> packed_output_tensor;

  // We have to use typedefs or else the template invocation in the type
  // confuses the pre-processor (EXPECT_DEATH is a macro).
  typedef BConv2DOpModel<float, float, float> FP_BConv2DOpModel;
  typedef BConv2DOpModel<float, float, std::int32_t> Bitpacked_BConv2DOpModel;

  // Test if fused ReLu throws an error in combination with zero-padding
  EXPECT_DEATH(
      {
        FP_BConv2DOpModel m_lce(
            compute_engine::tflite::Register_BCONV_2D_64_OPT, input_tensor,
            packed_filter_tensor, output_tensor, post_tensor, post_tensor,
            threshold_tensor, 64, 1, 1, Padding_SAME, 0,
            ActivationFunctionType_RELU, 1, 1, 1);
      },
      "Fused activations are only supported with valid or one-padding.");

  // Test if writing bitpacked output throws an error in combination with
  // zero-padding.
  EXPECT_DEATH(
      {
        Bitpacked_BConv2DOpModel m_lce(
            compute_engine::tflite::Register_BCONV_2D_64_OPT, input_tensor,
            packed_filter_tensor, packed_output_tensor, post_tensor,
            post_tensor, threshold_tensor, 64, 1, 1, Padding_SAME, 0,
            ActivationFunctionType_NONE, 1, 1, 1);
      },
      "Writing bitpacked output is only supported with valid or one-padding.");
}

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
