#include <gmock/gmock.h>

#include <numeric>

#include "larq_compute_engine/core/im2col_functor.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;

// In this test setup, a 1x5 kernel is used with a 5x5 input. Due to choice of
// the kernel size the expected output of im2col should be the transposed matrix
// of input
class Im2ColTestTranskernelCHW : public ::testing::Test {
 public:
  void SetUp() override {
    // initialize image input array with ascending numbers starting from 1
    data_im_expected.resize(im_num_elem);
    std::iota(std::begin(data_im_expected), std::end(data_im_expected), 1);

    // initialize the corresponding im2col array.
    data_col_expected.resize(col_num_elem);
    data_col_expected = {1,  6,  11, 16, 21, 2,  7,  12, 17, 22, 3,  8, 13,
                         18, 23, 4,  9,  14, 19, 24, 5,  10, 15, 20, 25};
  }

  // void TearDown() override {}

  const int num_channels = 1;
  const int im_h = 5, im_w = 5;
  const int kernel_h = 1, kernel_w = 5;
  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;
  const int im_num_elem = num_channels * im_h * im_w;
  const int col_num_elem = im_num_elem;

  std::vector<int> data_im_expected;
  std::vector<int> data_col_expected;
};

// In this test setup, a 2x2 kernel is used with a 4x4 input
class Im2ColTest2x2KernelCHW : public ::testing::Test {
 public:
  void SetUp() override {
    // initialize image input array with ascending numbers starting from 1
    data_im_expected.resize(im_num_elem);
    std::iota(std::begin(data_im_expected), std::end(data_im_expected), 1);

    // initialize the corresponding im2col array
    data_col_expected.resize(col_num_elem);
    data_col_expected = {
        1,  2,  3,  5,  6,  7,  9,  10, 11, 2,  3,  4,  6,  7,  8,  10, 11, 12,
        5,  6,  7,  9,  10, 11, 13, 14, 15, 6,  7,  8,  10, 11, 12, 14, 15, 16,
        17, 18, 19, 21, 22, 23, 25, 26, 27, 18, 19, 20, 22, 23, 24, 26, 27, 28,
        21, 22, 23, 25, 26, 27, 29, 30, 31, 22, 23, 24, 26, 27, 28, 30, 31, 32};
  }

  // void TearDown() override {}

  const int num_channels = 2;
  const int im_h = 4, im_w = 4;
  const int kernel_h = 2, kernel_w = 2;
  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;
  const int im_num_elem = num_channels * im_h * im_w;
  const int col_num_elem = 72;

  std::vector<int> data_im_expected;
  std::vector<int> data_col_expected;
};

class Im2ColTest3x3KernelwithPaddingHWC : public ::testing::Test {
 public:
  void SetUp() override {
    // initialize image input array with ascending numbers starting from 0
    data_im_expected.resize(im_num_elem);
    std::iota(std::begin(data_im_expected), std::end(data_im_expected), 0);

    // initialize the corresponding im2col array
    data_col_expected.resize(col_num_elem);
    data_col_expected = {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  0,  0,  8,  9,  10, 11,
        0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  8,  9,  10, 11, 12, 13,
        0,  0,  0,  0,  0,  0,  2,  3,  4,  5,  6,  7,  10, 11, 12, 13, 14, 15,
        0,  0,  0,  0,  0,  0,  4,  5,  6,  7,  0,  0,  12, 13, 14, 15, 0,  0,
        0,  0,  0,  1,  2,  3,  0,  0,  8,  9,  10, 11, 0,  0,  16, 17, 18, 19,
        0,  1,  2,  3,  4,  5,  8,  9,  10, 11, 12, 13, 16, 17, 18, 19, 20, 21,
        2,  3,  4,  5,  6,  7,  10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23,
        4,  5,  6,  7,  0,  0,  12, 13, 14, 15, 0,  0,  20, 21, 22, 23, 0,  0,
        0,  0,  8,  9,  10, 11, 0,  0,  16, 17, 18, 19, 0,  0,  24, 25, 26, 27,
        8,  9,  10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29,
        10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31,
        12, 13, 14, 15, 0,  0,  20, 21, 22, 23, 0,  0,  28, 29, 30, 31, 0,  0,
        0,  0,  16, 17, 18, 19, 0,  0,  24, 25, 26, 27, 0,  0,  0,  0,  0,  0,
        16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 0,  0,  0,  0,  0,  0,
        18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 0,  0,  0,  0,  0,  0,
        20, 21, 22, 23, 0,  0,  28, 29, 30, 31, 0,  0,  0,  0,  0,  0,  0,  0};
  }

  // void TearDown() override {}

  const int num_channels = 2;
  const int im_h = 4, im_w = 4;
  const int kernel_h = 3, kernel_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;
  const int im_num_elem = num_channels * im_h * im_w;
  const int col_num_elem = 288;

  std::vector<int> data_im_expected;
  std::vector<int> data_col_expected;
};

template <class TIm2ColFixture, class TIm2ColFunctor>
void test_im2col(const TIm2ColFixture* fix) {
  std::vector<int> data_col_output(fix->col_num_elem);
  TIm2ColFunctor im2col_functor;
  im2col_functor(fix->data_im_expected.data(), fix->num_channels, fix->im_h,
                 fix->im_w, fix->kernel_h, fix->kernel_w, fix->pad_h,
                 fix->pad_w, fix->stride_h, fix->stride_w, fix->dilation_h,
                 fix->dilation_w, data_col_output.data());

  EXPECT_THAT(data_col_output,
              ::testing::ElementsAreArray(fix->data_col_expected));
}

TEST_F(Im2ColTestTranskernelCHW, RefIm2ColTestCHW) {
  using TIm2ColFunctor = ce::core::ReferenceIm2ColFunctorCHW<int>;
  test_im2col<Im2ColTestTranskernelCHW, TIm2ColFunctor>(this);
}

TEST_F(Im2ColTest2x2KernelCHW, RefIm2ColTestCHW) {
  using TIm2ColFunctor = ce::core::ReferenceIm2ColFunctorCHW<int>;
  test_im2col<Im2ColTest2x2KernelCHW, TIm2ColFunctor>(this);
}

TEST_F(Im2ColTest3x3KernelwithPaddingHWC, RefIm2ColTestHWC) {
  using TIm2ColFunctor = ce::core::ReferenceIm2ColFunctorHWC<int>;
  test_im2col<Im2ColTest3x3KernelwithPaddingHWC, TIm2ColFunctor>(this);
}

}  // end namespace testing
}  // end namespace compute_engine
