"""Benchmarks for compute engine ops."""
import numpy as np
import tensorflow as tf
import larq_compute_engine as lqce
import time
import json

filename = "benchmark_results.json"
test_ops = [lqce.bsign, tf.sign]
test_dtypes = [np.int32, np.int64, np.float32, np.float64]
test_maxsize = 10000000
test_size_steps = 10


# Returns time in seconds (float)
def run_benchmark(operation, dtype, inputshape, numlayers):
    # Construct a random input tensor
    x = np.random.random_integers(-10, 10, size=inputshape).astype(dtype)
    # Construct the tensorflow graph
    # Potentially use multiple layers to avoid possible tensorflow startup overhead
    for i in range(numlayers):
        x = operation(x)
    # Evaluate the graph and time it
    with tf.compat.v1.Session() as sess:
        start = time.perf_counter()
        x.eval()
        end = time.perf_counter()

    return end - start


test_sizes = range(0, test_maxsize + 1, test_maxsize // test_size_steps)
results = []
for op in test_ops:
    for dtype in test_dtypes:
        print("Benchmarking {} - {}".format(op.__name__, dtype.__name__))
        subresults = []
        for s in test_sizes:
            # Reshaping can affect the results
            # shape = (s // 1000, 1000)
            shape = s
            result = run_benchmark(op, dtype, shape, 1)
            subresults += [(s, result)]
        results += [[op.__name__, dtype.__name__, subresults]]

print("Saving results to {}".format(filename))
with open(filename, "w") as filehandle:
    json.dump(results, filehandle)
