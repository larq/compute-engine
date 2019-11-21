#include <benchmark/benchmark.h>

#include <vector>

#include "larq_compute_engine/cc/core/aarch64/packbits.h"
#include "larq_compute_engine/cc/core/packbits.h"

using namespace compute_engine::core;
using namespace compute_engine::benchmarking;

void packbits_lqce(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  std::vector<float> in(in_size);
  std::vector<uint64_t> out(out_size);

  for (auto& x : in) x = -1.2345f;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    packbits_array<float, uint64_t>(&in[0], in_size, &out[0]);
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(&in[0]) % 64);
}

template <uint64_t (*packing_func)(float*)>
static void pack_wrapper(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  std::vector<float> in(in_size);
  std::vector<uint64_t> out(out_size);

  for (auto& x : in) x = -1.2345f;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    float* in_ptr = &in[0];
    for (auto i = 0u; i < out_size; ++i) {
      out[i] = packing_func(in_ptr);
      in_ptr += 64;
    }
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(&in[0]) % 64);
}

void packbits_64blocks(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  std::vector<float> in(in_size);
  std::vector<uint64_t> out(out_size);

  for (auto& x : in) x = -1.2345f;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    packbits_aarch64_64(&in[0], in_size / 64, &out[0]);
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(&in[0]) % 64);
}

void packbits_128blocks(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  std::vector<float> in(in_size);
  std::vector<uint64_t> out(out_size);

  for (auto& x : in) x = -1.2345f;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    packbits_aarch64_128(&in[0], in_size / 128, &out[0]);
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(&in[0]) % 64);
}

void packbits_dabnn(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  std::vector<float> in(in_size);
  std::vector<uint64_t> out(out_size);

  for (auto& x : in) x = -1.2345f;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    packbits_dabnn_128(&in[0], in_size / 128, &out[0]);
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(&in[0]) % 64);
}

constexpr size_t min_size = 128;
constexpr size_t max_size = 1 << 16;  // 64K
BENCHMARK(packbits_lqce)->Range(min_size, max_size);
BENCHMARK(packbits_64blocks)->Range(min_size, max_size);
BENCHMARK(packbits_128blocks)->Range(min_size, max_size);
BENCHMARK(packbits_dabnn)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(pack_wrapper, pack64bits_v1)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(pack_wrapper, pack64bits_v2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(pack_wrapper, pack64bits_v3)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(pack_wrapper, pack64bits_v4)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(pack_wrapper, pack64bits_v5)->Range(min_size, max_size);
