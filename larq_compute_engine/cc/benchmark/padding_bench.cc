#include <benchmark/benchmark.h>

#include "larq_compute_engine/cc/core/padding_functor.h"

using namespace compute_engine::core;

// Arguments: (input_size, filter_size, channels_in, channels_out)
template <typename T>
static void padding(benchmark::State& state) {
  const int input_height = state.range(0);
  const int input_width = state.range(0);
  const int filter_width = state.range(1);
  const int filter_height = state.range(1);
  const int channels_in = state.range(2);
  const int channels_out = state.range(3);

  const int input_batch_count = 1;

  const int filters_num_elem =
      filter_height * filter_width * channels_in * channels_out;

  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;

  const int output_height =
      (input_height - filter_height + 2 * pad_h) / stride_h + 1;
  const int output_width =
      (input_width - filter_width + 2 * pad_w) / stride_w + 1;
  const int output_num_elem =
      input_batch_count * output_height * output_width * channels_out;

  std::vector<T> filters_data;
  filters_data.resize(filters_num_elem);
  std::fill(std::begin(filters_data), std::end(filters_data), 1);

  using PaddingFunctor = ReferencePaddingFunctor<T, T, FilterFormat::OHWI>;

  std::vector<T> output;
  output.resize(output_num_elem);

  PaddingFunctor padding_functor;
  for (auto _ : state) {
    padding_functor(input_batch_count, input_height, input_width, channels_in,
                    filters_data.data(), filter_height, filter_width,
                    channels_out, stride_h, stride_w, output.data(),
                    output_height, output_width);
  }
}

BENCHMARK_TEMPLATE(padding, float)
    ->Ranges({{16, 32}, {1, 3}, {64, 128}, {16, 32}});
BENCHMARK_TEMPLATE(padding, float)
    ->Ranges({{16, 32}, {1, 3}, {64, 128}, {16, 32}});
BENCHMARK_TEMPLATE(padding, float)
    ->Ranges({{16, 32}, {1, 3}, {64, 128}, {16, 32}});
