// Note: make sure to disable cpu frequency scaling before testing these!
//      sudo cpupower frequency-set --governor performance
//      ./benchmark_aarch64
//      sudo cpupower frequency-set --governor powersave

#include <benchmark/benchmark.h>

#include <vector>

#include "larq_compute_engine/cc/core/aarch64/packbits.h"
#include "larq_compute_engine/cc/core/packbits.h"

using namespace compute_engine::core;

template <void (*packing_func)(const float*, size_t, uint64_t*),
          size_t blocksize>
static void packbits_bench(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  const size_t num_blocks = in_size / blocksize;

  std::vector<float> in(in_size, -1.0f);
  std::vector<uint64_t> out(out_size);

  for (auto _ : state) {
    benchmark::DoNotOptimize(out.data());
    packing_func(in.data(), num_blocks, out.data());
    benchmark::ClobberMemory();  // Force out data to be written to memory
  }
  // Print memory alignment to check if that affects results
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(in.data()) % 64);
}

constexpr size_t min_size = 128;
constexpr size_t max_size = 1 << 18;  // 256K
BENCHMARK_TEMPLATE(packbits_bench, packbits_array<float, uint64_t>, 1)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(packbits_bench, packbits_aarch64_64, 64)
    ->Range(min_size, max_size);
