#include <benchmark/benchmark.h>

#include "larq_compute_engine/cc/core/bconv2d_functor.h"

namespace ce = compute_engine;

static void bconv2d(benchmark::State& state) {
  using T = float;
  using TBitpacked = uint32_t;

  // We have to choose realistic sizes here for an honest benchmark
  const int input_depth = state.range(0);
  const int filter_count = input_depth;
  const int filter_width = 3;
  const int filter_height = 3;
  const int input_height = 32;
  const int input_width = 32;

  const int input_batch_count = 1;
  const int input_num_elem =
      input_batch_count * input_height * input_width * input_depth;

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

  std::vector<T> input_data;
  input_data.resize(input_num_elem);
  std::fill(std::begin(input_data), std::end(input_data), 1);

  std::vector<T> filters_data;
  filters_data.resize(filters_num_elem);
  std::fill(std::begin(filters_data), std::end(filters_data), 1);

  using TBGemmFunctor =
      ce::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;
  using TFusedBGemmFunctor =
      ce::core::FusedBGemmFunctor<T, T, T, TBitpacked, TBGemmFunctor>;
  using TBConv2DFunctor =
      ce::core::Im2ColBConvFunctor<T, T, T, TFusedBGemmFunctor>;

  std::vector<T> output;
  output.resize(output_num_elem);

  TBConv2DFunctor bconv2d_functor;
  for (auto _ : state) {
    bconv2d_functor(input_data.data(), input_batch_count, input_height,
                    input_width, input_depth, filters_data.data(), filter_height,
                    filter_width, filter_count, stride_h, stride_w,
                    1,  // pad_h, pad_w,
                    output.data(), output_height, output_width);
  }
}

BENCHMARK(bconv2d)->Arg(64)->Arg(256);
