#include <random>

#include "larq_compute_engine/core/bitpack_utils.h"
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/tests/utils.h"
#include "tensorflow/lite/kernels/test_util.h"

using namespace tflite;
using namespace compute_engine::core;

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_QUANTIZE();
TfLiteRegistration* Register_DEQUANTIZE();

namespace testing {

typedef TfLiteRegistration* (*register_function)(void);

class QuantizationOpModel : public SingleOpModel {
 public:
  QuantizationOpModel(register_function registration, const TensorData& input,
                      const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    // It doesn't matter if the name is not correct
    SetCustomOp("LceQuantization", {}, registration);
    BuildInterpreter({GetShape(input_)}, 1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }

 protected:
  int input_;
  int output_;
};

template <typename TInput>
class QuantizeOpModel : public QuantizationOpModel {
 public:
  using QuantizationOpModel::QuantizationOpModel;

  void SetInput(const std::vector<TInput>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<TBitpacked> GetOutput() {
    return ExtractVector<TBitpacked>(output_);
  }
};

template <typename TOutput>
class DequantizeOpModel : public QuantizationOpModel {
 public:
  using QuantizationOpModel::QuantizationOpModel;

  void SetInput(const std::vector<TBitpacked>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<TOutput> GetOutput() { return ExtractVector<TOutput>(output_); }
};

typedef std::tuple<std::array<int, 4>> TestParamTuple;

std::string TestNameSuffix(
    const ::testing::TestParamInfo<TestParamTuple>& info) {
  std::array<int, 4> shape = ::testing::get<0>(info.param);
  std::ostringstream oss;
  oss << "Shape" << shape[0] << "x" << shape[1] << "x" << shape[2] << "x"
      << shape[3];
  return oss.str();
}

template <typename TUnpacked>
void TestQuantization(const TestParamTuple& param) {
  std::array<int, 4> shape = ::testing::get<0>(param);

  int packed_channels = GetPackedSize(shape[3]);

  LceTensor<TUnpacked> unpacked_tensor(
      {shape[0], shape[1], shape[2], shape[3]});

  LceTensor<TBitpacked> packed_tensor(
      {shape[0], shape[1], shape[2], packed_channels});

  std::vector<TUnpacked> unpacked_data(shape[0] * shape[1] * shape[2] *
                                       shape[3]);

  std::random_device rd;
  std::mt19937 gen(rd());
  if (std::is_same<TUnpacked, std::int8_t>::value) {
    unpacked_tensor.GenerateQuantizationParams(gen);
  }
  unpacked_tensor.GenerateSigns(gen, std::begin(unpacked_data),
                                std::end(unpacked_data));

  QuantizeOpModel<TUnpacked> quantize_op(Register_QUANTIZE, unpacked_tensor,
                                         packed_tensor);
  quantize_op.SetInput(unpacked_data);
  quantize_op.Invoke();

  std::vector<TBitpacked> packed_data = quantize_op.GetOutput();

  DequantizeOpModel<TUnpacked> dequantize_op(Register_DEQUANTIZE, packed_tensor,
                                             unpacked_tensor);
  dequantize_op.SetInput(packed_data);
  dequantize_op.Invoke();

  EXPECT_EQ(dequantize_op.GetOutput(), unpacked_data);
}

class QuantizationOpTest : public ::testing::TestWithParam<TestParamTuple> {};

TEST_P(QuantizationOpTest, Float) { TestQuantization<float>(GetParam()); }

TEST_P(QuantizationOpTest, Int8) { TestQuantization<std::int8_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(AllCombinations, QuantizationOpTest,
                         ::testing::Values(std::array<int, 4>{1, 1, 1, 1},
                                           std::array<int, 4>{1, 4, 4, 1},
                                           std::array<int, 4>{1, 4, 4, 2},
                                           std::array<int, 4>{1, 4, 4, 31},
                                           std::array<int, 4>{1, 4, 4, 32},
                                           std::array<int, 4>{1, 4, 4, 33},
                                           std::array<int, 4>{1, 4, 4, 64},
                                           std::array<int, 4>{1, 4, 4, 68}),
                         TestNameSuffix);

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
