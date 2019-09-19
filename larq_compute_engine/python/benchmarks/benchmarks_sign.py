import tensorflow as tf
import larq_compute_engine as lqce
from benchmark import Benchmark


bsign_bench = Benchmark(lqce.bsign)
tfsign_bench = Benchmark(tf.sign)

bsign_bench.run_benchmark()
tfsign_bench.run_benchmark()

filename = "benchmark_results/benchmark_results_sign.json"

print("Saving results to {}".format(filename))

bsign_bench.save_results(filename, append=False)
tfsign_bench.save_results(filename, append=True)

print("Use plot_benchmarks.ipynb to view the results.")
