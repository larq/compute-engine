#ifndef LARQ_COMPUTE_ENGINE_TFLITE_TESTS_BCONV2D_OP
#define LARQ_COMPUTE_ENGINE_TFLITE_TESTS_BCONV2D_OP

#include <vector>

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "larq_compute_engine/core/types.h"
#include "larq_compute_engine/tflite/tests/utils.h"
#include "tensorflow/lite/kernels/test_util.h"

using namespace tflite;

namespace compute_engine {
namespace tflite {

TfLiteRegistration* Register_BCONV_2D_OPT();

namespace testing {

using compute_engine::core::TBitpacked;

typedef TfLiteRegistration* (*register_function)(void);

class BaseBConv2DOpModel : public SingleOpModel {
 public:
  BaseBConv2DOpModel(
      register_function registration, const TensorData& input,
      const TensorData& filter, const TensorData& output,
      const TensorData& post_activation_multiplier,
      const TensorData& post_activation_bias, const TensorData& thresholds,
      int channels_in, int stride_width = 1, int stride_height = 1,
      enum Padding padding = Padding_VALID, int pad_values = 0,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      int num_threads = -1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    post_activation_multiplier_ = AddInput(post_activation_multiplier);
    post_activation_bias_ = AddInput(post_activation_bias);
    thresholds_ = AddInput(thresholds);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("channels_in", channels_in);
      fbb.Int("stride_height", stride_height);
      fbb.Int("stride_width", stride_width);
      fbb.Int("dilation_height_factor", dilation_height_factor);
      fbb.Int("dilation_width_factor", dilation_width_factor);
      fbb.Int("padding", (int)padding);
      fbb.Int("pad_values", pad_values);
      fbb.Int("fused_activation_function", (int)activation);
    });
    fbb.Finish();
    SetCustomOp("LceBconv2d", fbb.GetBuffer(), registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_)}, num_threads,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }

 protected:
  int input_;
  int filter_;
  int output_;
  int post_activation_multiplier_;
  int post_activation_bias_;
  int thresholds_;
};

template <typename PostType, typename TOutput>
class BConv2DOpModel : public BaseBConv2DOpModel {
 public:
  using BaseBConv2DOpModel::BaseBConv2DOpModel;

  void SetFilter(const std::vector<TBitpacked>& f) {
    PopulateTensor(filter_, f);
  }

  void SetInput(const std::vector<TBitpacked>& data) {
    PopulateTensor(input_, data);
  }

  void SetPostActivationMultiplier(const std::vector<PostType>& f) {
    PopulateTensor(post_activation_multiplier_, f);
  }

  void SetPostActivationBias(const std::vector<PostType>& f) {
    PopulateTensor(post_activation_bias_, f);
  }

  void SetThresholds(const std::vector<std::int32_t>& f) {
    PopulateTensor(thresholds_, f);
  }

  std::vector<TOutput> GetOutput() { return ExtractVector<TOutput>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine

#endif  // LARQ_COMPUTE_ENGINE_TFLITE_TESTS_BCONV2D_OP
