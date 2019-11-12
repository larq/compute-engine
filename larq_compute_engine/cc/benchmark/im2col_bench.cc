#include <benchmark/benchmark.h>

#include <numeric>
#include <vector>

#include "larq_compute_engine/cc/core/im2col_functor.h"

static void im2col(benchmark::State& state) {
  const int im_h = state.range(0), im_w = state.range(0);

  const int num_channels = 64;
  const int kernel_h = 3, kernel_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;
  const int im_num_elem = num_channels * im_h * im_w;

  // Leave some extra space for padding
  const int col_num_elem =
      (im_h + 2) * (im_w + 2) * kernel_h * kernel_w * num_channels;

  std::vector<int> data_im_input(im_num_elem);
  std::vector<int> data_col_output(col_num_elem);

  // initialize image input array with ascending numbers starting from 0
  std::iota(std::begin(data_im_input), std::end(data_im_input), 0);

  compute_engine::core::ReferenceIm2ColFunctorHWC<int> im2col_functor;
  for (auto _ : state) {
    // This code gets timed
    im2col_functor(data_im_input.data(), num_channels, im_h, im_w, kernel_h,
                   kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                   dilation_w, data_col_output.data());
  }
}

BENCHMARK(im2col)->Arg(8)->Arg(256);
