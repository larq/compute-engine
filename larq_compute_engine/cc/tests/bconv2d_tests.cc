#include <gmock/gmock.h>
#include <cstdint>
#include <numeric>

#include "larq_compute_engine/cc/core/bconv2d_functor.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

template <class T, class TBitpacked>
void test_bconv2d_one_filter(const int input_height, const int input_width,
                             const int input_depth, const int filter_height,
                             const int filter_width) {
  const int input_batch_count = 1;
  const int input_num_elem =
      input_batch_count * input_height * input_width * input_depth;

  const int filter_count = 1;
  const int filters_num_elem =
      filter_height * filter_width * input_depth * filter_count;

  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;

  const int output_height =
      (input_height - filter_height + 2 * pad_h) / stride_h + 1;
  const int output_width =
      (input_width - filter_width + 2 * pad_w) / stride_w + 1;
  const int output_num_elem =
      input_batch_count * output_height * output_width * filter_count;

  const int output_expected_value = input_depth * filter_height * filter_width;

  std::vector<T> input_data;
  input_data.resize(input_num_elem);
  std::fill(std::begin(input_data), std::end(input_data), 1);

  std::vector<T> filters_data;
  filters_data.resize(filters_num_elem);
  std::fill(std::begin(filters_data), std::end(filters_data), 1);

  std::vector<T> output_expected;
  output_expected.resize(output_num_elem);
  std::fill(std::begin(output_expected), std::end(output_expected),
            output_expected_value);

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, T, T, TBitpacked, TBGemmFunctor>;
  using TBConv2DFunctor =
      ce::core::Im2ColBConvFunctor<T, T, T, TFusedBGemmFunctor>;

  std::vector<T> output;
  output.resize(output_num_elem);

  TBConv2DFunctor bconv2d_functor;
  bconv2d_functor(input_data.data(), input_batch_count, input_height,
                  input_width, input_depth, filters_data.data(), filter_height,
                  filter_width, filter_count, stride_h, stride_w,
                  1,  // pad_h, pad_w,
                  output.data(), output_height, output_width);

  EXPECT_THAT(output, ::testing::ElementsAreArray(output_expected));
}

TEST(BConv2DOneFilterTests, I4x4F1x1Uint8) {
  test_bconv2d_one_filter<float, std::uint8_t>(4, 4, 1, 1, 1);
}

TEST(BConv2DOneFilterTests, I4x4F1x1Uint32) {
  test_bconv2d_one_filter<float, std::uint32_t>(4, 4, 1, 1, 1);
}

TEST(BConv2DOneFilterTests, I4x4F1x1Uint64) {
  test_bconv2d_one_filter<float, std::uint64_t>(4, 4, 1, 1, 1);
}

TEST(BConv2DOneFilterTests, I4x4F3x3Uint8) {
  test_bconv2d_one_filter<float, std::uint8_t>(4, 4, 1, 3, 3);
}

TEST(BConv2DOneFilterTests, I4x4F3x3Uint32) {
  test_bconv2d_one_filter<float, std::uint32_t>(4, 4, 1, 3, 3);
}

TEST(BConv2DOneFilterTests, I4x4F3x3Uint64) {
  test_bconv2d_one_filter<float, std::uint64_t>(4, 4, 1, 3, 3);
}

}  // end namespace testing
}  // end namespace compute_engine
