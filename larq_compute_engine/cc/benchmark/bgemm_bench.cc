#include <benchmark/benchmark.h>
#include <vector>

#include "larq_compute_engine/cc/core/bgemm_functor.h"
#include "larq_compute_engine/cc/core/fused_bgemm_functor.h"

static void bgemm(benchmark::State& state) {
  using TIn = uint64_t;
  using TOut = int32_t;
  using BGemmFunctor = compute_engine::core::ReferenceBGemmFunctor<TIn, TIn, TOut>;

  // Let us use realistic sizes here for an honest benchmark
  //
  // a is input, after being transformed by im2col
  // b is the filter of size output_channels x (3 x 3 x input_channels)

  const int in_channels = (state.range(0) + 63) / 64; // elements *after* bitpacking
  const int out_channels = in_channels;
  const int imagesize = 32 * 32;

  const int m = imagesize;
  const int n = out_channels;
  const int k = 3 * 3 * in_channels;

  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::vector<TIn> a(a_size, 1);
  std::vector<TIn> b(b_size, 1);
  std::vector<TOut> c(c_size);

  BGemmFunctor bgemm_functor;
  for (auto _ : state) {
    bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldc);
  }
}

static void fused_bgemm(benchmark::State& state) {
  using TBitpacked = uint8_t;
  using T = float;

  const int in_channels = state.range(0);
  const int out_channels = in_channels;
  const int imagesize = 32 * 32;

  const int m = imagesize;
  const int n = out_channels;
  const int k = 3 * 3 * in_channels;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::vector<T> a_data_float(a_size, 1.0f);
  std::vector<T> b_data_float(b_size, -1.0f);
  std::vector<T> c(c_size);

  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  using TBGemmFunctor =
      compute_engine::core::ReferenceBGemmFunctor<TBitpacked, TBitpacked, T>;
  using TFusedBGemmFunctor =
      compute_engine::core::FusedBGemmFunctor<T, T, T, TBitpacked, TBGemmFunctor>;

  TFusedBGemmFunctor fused_bgemm_functor;
  for (auto _ : state) {
    fused_bgemm_functor(m, n, k, a_data_float.data(), lda, b_data_float.data(),
                        ldb, c.data(), ldc);
  }
}

BENCHMARK(bgemm)->Arg(64)->Arg(256);
BENCHMARK(fused_bgemm)->Arg(64)->Arg(256);
