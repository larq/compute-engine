"""Benchmarks for compute engine ops."""
import numpy as np
import tensorflow as tf
import time
import datetime
import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Benchmark:
    """Benchmark for a tensorflow operation.

    Use `run_benchmark()` to run the benchmark on all datatypes and input sizes. This will internally store a list of results.
    Use `save_results(filename, append)` to store the results to a JSON file. If append is true, it will assume the file already contains other benchmark results and append the new results.

    The file `plot_benchmarks.ipynb` can be used to process the files and plot the results.

    # Arguments
    op: The operator to test, for example `tf.sign`.
    dtypes: List of data types that should be tested. By default it will test all datatypes and automatically ignore the types that are not supported by the operator.
    data_shape: When set to a tuple `(d1,...,dk)` then the inputs to the operator will be of shapes `(s,d1,...,dk)` where s varies from `0` to `max_inputsize`.  If set to `None` (default) then the input tensor will always be flat, of shape (s).
    max_inputsize: Maximum size of input tensor. When input tensor has shape `(s,d1,...,dk)` then `s` will be limited such that `s * d1 * ... * dk <= max_inputsize`.
    inputsize_steps: The benchmark will try this many sizes from 0 up to max_inputsize.
    repeat: The number of times to repeat each individual measurement for accuracy. The average will be stored.

    !!! example
        ```python
        sign_benchmark = Benchmark(tf.sign)
        sign_benchmark.run_benchmark()
        sign_benchmark.save_results("benchmark_results_sign.json", append = True)
        ```
    !!! note
        For every data type that is tested, the JSON file contains an entry of which `entry['timings']` gives an array of pairs `(size, time)` where `size` is the total number of elements in the input tensor and `time` is the time in seconds that the benchmark took.
    """

    def __init__(
        self,
        op,
        dtypes=[np.int8, np.int32, np.int64, np.float32, np.float64],
        data_shape=None,
        max_inputsize=4 * 1024 * 1024,
        inputsize_steps=10,
        repeat=4,
    ):
        self.op = op
        self.dtypes = dtypes
        self.max_inputsize = max_inputsize
        self.repeat = repeat
        self.results = []
        if data_shape is None:
            self.shape_string = "(-1)"
            max_s = max_inputsize
            s_range = range(0, max_s + 1, int(max_s / inputsize_steps))
            self.test_shapes = [(s,) for s in s_range]
        else:
            self.shape_string = str((-1,) + data_shape)
            max_s = max_inputsize // np.prod(data_shape)
            if max_s < 1:
                raise ValueError(
                    "max_inputsize {} is smaller than data_shape {}".format(
                        max_inputsize, data_shape
                    )
                )
            s_range = range(0, max_s + 1, int(np.ceil(max_s / inputsize_steps)))
            self.test_shapes = [(s,) + data_shape for s in s_range]

    def run_benchmark(self):
        self.results = []
        for dtype in self.dtypes:
            log.info("Benchmarking {} - {}".format(self.op.__name__, dtype.__name__))
            try:
                timeresults = []
                for shape in self.test_shapes:
                    result = 0
                    for _ in range(self.repeat):
                        result += self._single_run(shape, dtype)
                    result /= self.repeat
                    timeresults += [(int(np.prod(shape)), result)]
                self.results.append(
                    {
                        "datetime": str(datetime.datetime.now()),
                        "operation": self.op.__name__,
                        "data_type": dtype.__name__,
                        "data_shape": self.shape_string,
                        "timings": timeresults,
                    }
                )
            except TypeError:
                log.info(
                    "Operator {} does not support {}.".format(
                        self.op.__name__, dtype.__name__
                    )
                )

    def _single_run(self, inputshape, dtype):
        """Runs the op once and returns the elapsed time in seconds.
        """
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

        results.extend(self.results)
        with open(filename, "w") as filehandle:
            json.dump(results, filehandle)
