"""Benchmarks for compute engine ops."""
import numpy as np
import tensorflow as tf
import time
import datetime
import json
import logging

log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark configuration for a tensorflow operation.

    Use `run_benchmark()` to run the benchmark on all datatypes and input sizes. This will internally store a list of results.
    Use `save_results(filename, append)` to store the results to a JSON file. If append is true, it will assume the file already contains other benchmark results and append the new results.

    The file `plot_benchmarks.ipynb` can be used to process the files and plot the results.

    # Arguments
    op: The operator to test, for example `tf.sign`.
    dtypes: List of data types that should be tested. By default it will test all datatypes and automatically ignore the types that are not supported by the operator.
    max_inputsize: Maximum size of input tensor that should be given to the operator.
    inputsize_steps: The benchmark will try this many sizes from 0 up to max_inputsize.

    !!! example
        ```python
        sign_benchmark = Benchmark(tf.sign)
        sign_benchmark.run_benchmark()
        sign_benchmark.save_results("benchmark_results_sign.json", append = True)
        ```
    """

    # TODO: Add option for different shapes
    # because input shapes affect the results, even for simple
    # operations like component-wise sign.
    def __init__(
        self,
        op,
        dtypes=[np.int8, np.int32, np.int64, np.float32, np.float64, np.complex],
        max_inputsize=1000000,
        inputsize_steps=10,
    ):
        self.op = op
        self.dtypes = dtypes
        self.max_inputsize = max_inputsize
        self.size_steps = inputsize_steps
        self.results = []

    def run_benchmark(self):
        test_sizes = range(
            0, self.max_inputsize + 1, self.max_inputsize // self.size_steps
        )

        self.results = []
        for dtype in self.dtypes:
            log.info("Benchmarking {} - {}".format(self.op.__name__, dtype.__name__))
            try:
                timeresults = []
                for s in test_sizes:
                    shape = s
                    result = self._single_run(shape, dtype)
                    timeresults += [(s, result)]
                self.results += [
                    {
                        "datetime": str(datetime.datetime.now()),
                        "operation": self.op.__name__,
                        "data_type": dtype.__name__,
                        "timings": timeresults,
                    }
                ]
            except TypeError:
                log.info(
                    "Operator {} does not support {}.".format(
                        self.op.__name__, dtype.__name__
                    )
                )

    def _single_run(self, inputshape, dtype):
        """Runs the op once and returns the elapsed time in seconds.
        """
        # Construct a random input tensor
        x = np.random.random_integers(-10, 10, size=inputshape).astype(dtype)
        # Construct the tensorflow graph
        # TODO: It might make sense to put the operation inbetween some often-used layers
        # to get a better idea of how it runs in a realistic tensorflow graph,
        # and avoid the startup cost of passing the input data to the graph and so on.
        # Just as input shapes can affect the result, the graph structure might as well.
        x = self.op(x)
        # Evaluate the graph and time it
        with tf.compat.v1.Session():
            start = time.perf_counter()
            x.eval()
            end = time.perf_counter()
        return end - start

    def save_results(self, filename, append=True):
        results = []
        if append:
            with open(filename, "r") as filehandle:
                results = json.load(filehandle)

        results.append(self.results)
        with open(filename, "w") as filehandle:
            json.dump(results, filehandle)
