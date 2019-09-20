import tensorflow as tf
import larq_compute_engine as lqce
from benchmark import Benchmark

bsign_bench_flat = Benchmark(lqce.bsign, data_shape=None)
bsign_bench = Benchmark(lqce.bsign, data_shape=(128, 128, 32))
tfsign_bench_flat = Benchmark(tf.sign, data_shape=None)
tfsign_bench = Benchmark(tf.sign, data_shape=(128, 128, 32))

bsign_bench_flat.run_benchmark()
bsign_bench.run_benchmark()
tfsign_bench_flat.run_benchmark()
tfsign_bench.run_benchmark()


filename = "benchmark_results/benchmark_results_sign.json"

print("Saving results to {}".format(filename))

bsign_bench_flat.save_results(filename, append=False)
bsign_bench.save_results(filename, append=True)
tfsign_bench_flat.save_results(filename, append=True)
tfsign_bench.save_results(filename, append=True)

print("Use plot_benchmarks.ipynb to view the results.")
