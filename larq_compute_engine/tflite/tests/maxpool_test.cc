#include <random>

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/packbits_utils.h"
#include "larq_compute_engine/tflite/tests/utils.h"
#include "tensorflow/lite/kernels/test_util.h"

using namespace tflite;
using namespace compute_engine::core;

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BMAXPOOL_2D();

namespace testing {

// We compare against TFLite's maxpool
// This is copied from pooling_test.cc
class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(
      BuiltinOperator type, const TensorData& input, int filter_width,
      int filter_height, const TensorData& output,
      Padding padding = Padding_VALID, int stride_w = 2, int stride_h = 2,
      ActivationFunctionType activation = ActivationFunctionType_NONE) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(type, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, stride_w, stride_h,
                                     filter_width, filter_height, activation)
                     .Union());

    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

template <typename T>
class PoolingOpModel : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::vector<T> data) { PopulateTensor(input_, data); }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

// We cannot use `uint32` here because the tflite tenor type requires int32
using TBitpacked = std::int32_t;

typedef TfLiteRegistration* (*register_function)(void);

class BaseBMaxPool2DOpModel : public SingleOpModel {
 public:
  BaseBMaxPool2DOpModel(register_function registration, const TensorData& input,
                        const TensorData& output, int filter_height = 2,
                        int filter_width = 2, int stride_height = 1,
                        int stride_width = 1,
                        enum Padding padding = Padding_SAME) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("stride_height", stride_height);
      fbb.Int("stride_width", stride_width);
      fbb.Int("filter_height", filter_height);
      fbb.Int("filter_width", filter_width);
      fbb.String("padding", GetPaddingName(padding));
    });
    fbb.Finish();
    SetCustomOp("LceBmaxpool2d", fbb.GetBuffer(), registration);
    BuildInterpreter({GetShape(input_)}, 1);
  }

 protected:
  int input_;
  int output_;
};

template <typename TInput>
class BMaxPool2DOpModel : public BaseBMaxPool2DOpModel {
 public:
  using BaseBMaxPool2DOpModel::BaseBMaxPool2DOpModel;

  void SetInput(const std::vector<TInput>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<TBitpacked> GetOutput() {
    return ExtractVector<TBitpacked>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

typedef std::tuple<std::array<int, 4>,  // input shape [BHWI]
                   std::array<int, 2>,  // filter shape [HW]
                   std::array<int, 2>,  // strides [HW]
                   Padding,             // paddding
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
        stride_height(::testing::get<2>(param_tuple)[0]),
        stride_width(::testing::get<2>(param_tuple)[1]),
        padding(::testing::get<3>(param_tuple)),
        kernel_name(::testing::get<4>(param_tuple).first),
        registration(::testing::get<4>(param_tuple).second) {}

  static std::string TestNameSuffix(
      const ::testing::TestParamInfo<TestParamTuple>& info) {
    const TestParam param(info.param);
    std::ostringstream oss;
    oss << "I" << param.input_batch_count << "x" << param.input_height << "x"
        << param.input_width << "x" << param.input_depth;
    oss << "_K" << param.filter_height << "x" << param.filter_width;
    oss << "_S" << param.stride_height << "x" << param.stride_width;
    oss << "_P" << GetPaddingName(param.padding);
    return oss.str();
  }

  int input_batch_count = 1;
  int input_height = 2;
  int input_width = 2;
  int input_depth = 1;

  int filter_height = 2;
  int filter_width = 2;

  int stride_height = 1;
  int stride_width = 1;

  Padding padding = Padding_VALID;

  std::string kernel_name = "Unknown";
  register_function registration = compute_engine::tflite::Register_BMAXPOOL_2D;
};

class BMaxPool2DOpTest : public ::testing::TestWithParam<TestParamTuple> {};

TEST_P(BMaxPool2DOpTest, FloatAndBinaryInput) {
  TestParam params(GetParam());

  int packed_input_depth = GetPackedSize<TBitpacked>(params.input_depth);

  LceTensor<float> input_tensor({params.input_batch_count, params.input_height,
                                 params.input_width, params.input_depth});
  LceTensor<TBitpacked> packed_input_tensor(
      {params.input_batch_count, params.input_height, params.input_width,
       packed_input_depth});

  LceTensor<float> output_tensor;
  LceTensor<TBitpacked> packed_output_tensor;

  RuntimeShape input_shape = GetShape(input_tensor.shape);
  RuntimeShape packed_input_shape = GetShape(packed_input_tensor.shape);

  std::vector<float> input_data;
  std::vector<TBitpacked> input_data_bp;

  input_data.resize(input_shape.FlatSize());
  input_data_bp.resize(packed_input_shape.FlatSize());

  std::random_device rd;
  std::mt19937 gen(rd());
  input_tensor.GenerateSigns(gen, std::begin(input_data), std::end(input_data));

  // Bitpack the input
  packbits_tensor<BitpackOrder::Canonical>(input_shape, input_data.data(), 0,
                                           packed_input_shape,
                                           input_data_bp.data());

  // Our op with binary input
  BMaxPool2DOpModel<TBitpacked> m_lce_binary(
      params.registration, packed_input_tensor, packed_output_tensor,
      params.filter_height, params.filter_width, params.stride_height,
      params.stride_width, params.padding);
  m_lce_binary.SetInput(input_data_bp);
  m_lce_binary.Invoke();

  // Our op with float input
  BMaxPool2DOpModel<float> m_lce_float(
      params.registration, input_tensor, packed_output_tensor,
      params.filter_height, params.filter_width, params.stride_height,
      params.stride_width, params.padding);
  m_lce_float.SetInput(input_data);
  m_lce_float.Invoke();

  // Builtin op with float input
  PoolingOpModel<float> m_builtin(BuiltinOperator_MAX_POOL_2D, input_tensor,
                                  params.filter_width, params.filter_height,
                                  output_tensor, params.padding,
                                  params.stride_width, params.stride_height);
  m_builtin.SetInput(input_data);
  m_builtin.Invoke();

  // Bitpack the tflite output
  RuntimeShape out_shape = GetShape(m_builtin.GetOutputShape());
  std::vector<TBitpacked> builtin_output_data_bp(
      GetPackedTensorSize<TBitpacked>(out_shape));
  RuntimeShape packed_out_shape;
  packbits_tensor<BitpackOrder::Canonical>(
      out_shape, m_builtin.GetOutput().data(), 0, packed_out_shape,
      builtin_output_data_bp.data());

  // Check our binary op
  EXPECT_EQ(m_lce_binary.GetOutputShape(), GetShape(packed_out_shape));
  EXPECT_EQ(m_lce_binary.GetOutput(), builtin_output_data_bp);

  // Check our float op
  EXPECT_EQ(m_lce_float.GetOutputShape(), GetShape(packed_out_shape));
  EXPECT_EQ(m_lce_float.GetOutput(), builtin_output_data_bp);
}

TEST_P(BMaxPool2DOpTest, Int8Input) {
  TestParam params(GetParam());

  int packed_input_depth = GetPackedSize<TBitpacked>(params.input_depth);

  LceTensor<std::int8_t> input_tensor({params.input_batch_count,
                                       params.input_height, params.input_width,
                                       params.input_depth});
  LceTensor<TBitpacked> packed_input_tensor(
      {params.input_batch_count, params.input_height, params.input_width,
       packed_input_depth});

  LceTensor<std::int8_t> output_tensor;
  LceTensor<TBitpacked> packed_output_tensor;

  RuntimeShape input_shape = GetShape(input_tensor.shape);
  RuntimeShape packed_input_shape = GetShape(packed_input_tensor.shape);

  std::vector<std::int8_t> input_data;
  std::vector<TBitpacked> input_data_bp;

  input_data.resize(input_shape.FlatSize());
  input_data_bp.resize(packed_input_shape.FlatSize());

  std::random_device rd;
  std::mt19937 gen(rd());
  input_tensor.GenerateQuantizationParams(gen);
  output_tensor.SetQuantizationParams(input_tensor);  // Required by tflite spec
  input_tensor.GenerateSigns(gen, std::begin(input_data), std::end(input_data));

  // Bitpack the input
  packbits_tensor<BitpackOrder::Canonical>(
      input_shape, input_data.data(), input_tensor.zero_point,
      packed_input_shape, input_data_bp.data());

  // Our op with int8 input
  BMaxPool2DOpModel<std::int8_t> m_lce(
      params.registration, input_tensor, packed_output_tensor,
      params.filter_height, params.filter_width, params.stride_height,
      params.stride_width, params.padding);
  m_lce.SetInput(input_data);
  m_lce.Invoke();

  // Builtin op with int8 input
  PoolingOpModel<std::int8_t> m_builtin(
      BuiltinOperator_MAX_POOL_2D, input_tensor, params.filter_width,
      params.filter_height, output_tensor, params.padding, params.stride_width,
      params.stride_height);
  m_builtin.SetInput(input_data);
  m_builtin.Invoke();

  // Bitpack the tflite output
  RuntimeShape out_shape = GetShape(m_builtin.GetOutputShape());
  std::vector<TBitpacked> builtin_output_data_bp(
      GetPackedTensorSize<TBitpacked>(out_shape));
  RuntimeShape packed_out_shape;
  packbits_tensor<BitpackOrder::Canonical>(
      out_shape, m_builtin.GetOutput().data(), output_tensor.zero_point,
      packed_out_shape, builtin_output_data_bp.data());

  EXPECT_EQ(m_lce.GetOutputShape(), GetShape(packed_out_shape));
  EXPECT_EQ(m_lce.GetOutput(), builtin_output_data_bp);
}

using ::testing::Values;
INSTANTIATE_TEST_SUITE_P(
    AllCombinations, BMaxPool2DOpTest,
    ::testing::Combine(
        Values(std::array<int, 4>{1, 7, 7, 1}, std::array<int, 4>{1, 8, 5, 1},
               std::array<int, 4>{2, 7, 7, 64}, std::array<int, 4>{2, 8, 5, 64},
               std::array<int, 4>{1, 7, 7, 130},
               std::array<int, 4>{1, 8, 5, 200}),  // input shape [BHWI]
        Values(std::array<int, 2>{1, 1}, std::array<int, 2>{2, 2},
               std::array<int, 2>{2, 3},
               std::array<int, 2>{3, 3}),  // filter shape [HWO]
        Values(std::array<int, 2>{1, 1}, std::array<int, 2>{2, 2},
               std::array<int, 2>{2, 3}),     // strides height/width
        Values(Padding_VALID, Padding_SAME),  // padding
        Values(std::make_pair("Ref", Register_BMAXPOOL_2D))),
    TestParam::TestNameSuffix);

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
