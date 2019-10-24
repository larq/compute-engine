#include <benchmark/benchmark.h>
#include <vector>

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "larq_compute_engine/cc/core/fused_bgemm_functor.h"

// Arguments are similar to the bconv2d one:
// (input_size, filter_size, channels_in, channels_out)
template <typename TIn, typename TOut>
static void bgemm(benchmark::State& state) {
  const int input_height = state.range(0);
  const int input_width = state.range(0);
  const int filter_width = state.range(1);
  const int filter_height = state.range(1);
  const int channels_in = state.range(2);
  const int channels_out = state.range(3);

  const int channels_in_bp = (channels_in + sizeof(TIn) - 1) / sizeof(TIn);

  // a is input, after being transformed by im2col
  //   So technically `m` should be `ouput_height * output_width`
  //   but let's assume `SAME` padding here
  // b is the filter of size output_channels x (3 x 3 x input_channels)

  const int m = input_height * input_width;
  const int n = channels_out;
  const int k = filter_height * filter_width * channels_in_bp;

  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::vector<TIn> a(a_size, 1);
  std::vector<TIn> b(b_size, 1);
  std::vector<TOut> c(c_size);

  using BGemmFunctor = compute_engine::core::ReferenceBGemmFunctor<TIn, TIn, TOut>;
  BGemmFunctor bgemm_functor;
  for (auto _ : state) {
    bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);
  }
}

template <typename TIn, typename TBitpacked, typename TOut>
static void fused_bgemm(benchmark::State& state) {
  const int input_height = state.range(0);
  const int input_width = state.range(0);
  const int filter_width = state.range(1);
  const int filter_height = state.range(1);
  const int channels_in = state.range(2);
  const int channels_out = state.range(3);

  const int m = input_height * input_width;
  const int n = channels_out;
  const int k = filter_height * filter_width * channels_in;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::vector<TIn> a_data_float(a_size, 1.0f);
  std::vector<TIn> b_data_float(b_size, -1.0f);
  std::vector<TOut> c(c_size);

  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  using TBGemmFunctor =
      compute_engine::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, TOut>;
  using TFusedBGemmFunctor =
      compute_engine::core::FusedBGemmFunctor<TIn, TIn, TOut, TBitpacked, TBGemmFunctor>;

  TFusedBGemmFunctor fused_bgemm_functor;
  for (auto _ : state) {
    fused_bgemm_functor(m, n, k, a_data_float.data(), lda, b_data_float.data(),
                        ldb, c.data(), ldc);
  }
}

BENCHMARK_TEMPLATE(bgemm, uint8_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});
BENCHMARK_TEMPLATE(bgemm, uint32_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});
BENCHMARK_TEMPLATE(bgemm, uint64_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});
BENCHMARK_TEMPLATE(fused_bgemm, float, uint8_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});
BENCHMARK_TEMPLATE(fused_bgemm, float, uint32_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});
BENCHMARK_TEMPLATE(fused_bgemm, float, uint64_t, float)->Ranges({{16,32},{3,5},{8,128},{4,16}});

