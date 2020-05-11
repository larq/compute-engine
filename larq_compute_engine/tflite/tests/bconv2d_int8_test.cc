#include <gtest/gtest.h>

#include "larq_compute_engine/tflite/tests/bconv2d_op_model.h"
#include "larq_compute_engine/tflite/tests/utils.h"

namespace compute_engine {
namespace tflite {
namespace testing {

TEST(BConv2DTests, Int8ErrorTest) {
  LceTensor<std::int8_t> input_tensor({1, 16, 16, 64});
  LceTensor<std::int32_t> packed_filter_tensor({128, 3, 3, 64});
  LceTensor<std::int8_t> post_tensor({128});
  LceTensor<std::int8_t> output_tensor;

  // Required for the macro
  typedef BConv2DOpModel<std::int8_t, std::int8_t, std::int8_t>
      Int8_BConv2DOpModel;

  EXPECT_DEATH(
      {
        Int8_BConv2DOpModel m_lce(
            compute_engine::tflite::Register_BCONV_2D64_OPT, input_tensor,
            packed_filter_tensor, output_tensor, post_tensor, post_tensor, 64,
            1, 1, Padding_SAME, 0, ActivationFunctionType_NONE, 1, 1, 1);
      },
      "8-bit quantization is only supported with valid or one-padding.");
}

TEST(BConv2DTests, Int8PostTest) {
  using T = std::int8_t;
  LceTensor<T> input_tensor({1, 2, 2, 2});
  LceTensor<std::int32_t> packed_filter_tensor({4, 2, 2, 2});
  LceTensor<T> post_tensor({4});
  LceTensor<T> output_tensor;

  input_tensor.scale = 1.0f;
  input_tensor.zero_point = 0;

  // This is the interesting part that we are testing
  post_tensor.scale = 1.0f / 16.0f;
  post_tensor.zero_point = 5;

  output_tensor.scale = 1.0f / 32.0f;
  output_tensor.zero_point = 0;

  BConv2DOpModel<T, T, T> m_lce(
      compute_engine::tflite::Register_BCONV_2D64_OPT, input_tensor,
      packed_filter_tensor, output_tensor, post_tensor, post_tensor, 2, 1, 1,
      Padding_VALID, 0, ActivationFunctionType_NONE, 1, 1, 1);

  m_lce.SetInput({
      1, 1,   // batch = 0, y = 0, x = 0
      1, -1,  // batch = 0, y = 0, x = 1
      1, 1,   // batch = 0, y = 1, x = 0
      1, -1   // batch = 0, y = 1, x = 1
  });
  // Bitpacked filter. Since we're only testing post_ stuff here, its easiest to
  // set all filter values to +1, which means bitpacked 0
  m_lce.SetFilter({
      0, 0,  // out channel = 0, y = 0, x = 0
      0, 0,  // out channel = 0, y = 0, x = 1
      0, 0,  // out channel = 0, y = 1, x = 0
      0, 0,  // out channel = 0, y = 1, x = 1
      0, 0,  // out channel = 1, y = 0, x = 0
      0, 0,  // out channel = 1, y = 0, x = 1
      0, 0,  // out channel = 1, y = 1, x = 0
      0, 0,  // out channel = 1, y = 1, x = 1
      0, 0,  // out channel = 2, y = 0, x = 0
      0, 0,  // out channel = 2, y = 0, x = 1
      0, 0,  // out channel = 2, y = 1, x = 0
      0, 0,  // out channel = 2, y = 1, x = 1
      0, 0,  // out channel = 3, y = 0, x = 0
      0, 0,  // out channel = 3, y = 0, x = 1
      0, 0,  // out channel = 3, y = 1, x = 0
      0, 0   // out channel = 3, y = 1, x = 1
  });

  // The real post values will be (x - 5) / 16
  m_lce.SetPostActivationMultiplier({
      5,   // out_channel = 0 -> 0
      21,  // out_channel = 1 -> 1
      4,   // out_channel = 2 -> -1/16
      7    // out_channel = 3 ->  1/8
  });
  m_lce.SetPostActivationBias({
      5,  // out_channel = 0 -> 0
      2,  // out_channel = 1 -> -3/16
      9,  // out_channel = 2 -> 1/4
      21  // out_channel = 3 -> 1
  });

  m_lce.Invoke();

  // There's only a single output pixel, with 4 output channels.
  // Before the post step the accumulator value is +4, for all output channels.
  // So we're doing
  // post_add + 4 * post_mul
  // Then the output scale is 1/32, so multiply by 32 to get the int8 value
  //
  // 0     + 4 * 0       = 0     -> 0
  // -3/16 + 4 * 1       = 61/16 -> 122
  // 1/4   + 4 * (-1/16) = 0     -> 0
  // 1     + 4 * (1/8)   = 3/2   -> 48
  EXPECT_THAT(m_lce.GetOutput(), ::testing::ElementsAreArray({0, 122, 0, 48}));
}

}  // namespace testing
}  // namespace tflite
}  // namespace compute_engine
