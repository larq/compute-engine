// Note: make sure to disable cpu frequency scaling before testing these!
//      sudo cpupower frequency-set --governor performance
//      ./benchmark_aarch64
//      sudo cpupower frequency-set --governor powersave

#include <benchmark/benchmark.h>

#include <cstdlib>

#include "larq_compute_engine/cc/core/aarch64/packbits.h"
#include "larq_compute_engine/cc/core/packbits.h"

using namespace compute_engine::core;

constexpr size_t min_size = 128;
constexpr size_t max_size = 1 << 16;  // 64K

// Benchmark results seem to depend on memory alignment,
// not only to cache-size (64 bytes) but also page size (4KB).
// To get a fair comparison, we test all functions on the same memory.
constexpr size_t alignment = 4 * 1024;
float* in_array = reinterpret_cast<float*>(
    aligned_alloc(alignment, max_size * sizeof(float)));
uint64_t* out_array = reinterpret_cast<uint64_t*>(
    aligned_alloc(alignment, (max_size / 64) * sizeof(uint64_t)));

template <void (*packing_func)(const float*, size_t, uint64_t*),
          size_t blocksize>
static void packbits_bench(benchmark::State& state) {
  const size_t in_size = state.range(0);
  const size_t out_size = in_size / 64;

  const size_t num_blocks = in_size / blocksize;

  for (auto _ : state) {
    benchmark::DoNotOptimize(out_array);
    packing_func(in_array, num_blocks, out_array);
    benchmark::ClobberMemory();  // Force output data to be written to memory
  }
  // Print memory alignment to check if that affects results
  state.counters["alignment"] = (reinterpret_cast<uint64_t>(in_array) % 4096);
}

BENCHMARK_TEMPLATE(packbits_bench, packbits_array<float, uint64_t>, 1)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(packbits_bench, packbits_aarch64_64, 64)
    ->Range(min_size, max_size);
