# Benchmarking TF Lite

## Building the benchmark program

See the [documentation](../../../docs/build.md) on how to configure bazel
and then build the `//larq_compute_engine/tflite/benchmark:lce_benchmark_model`
bazel target.

## Running the benchmark

Simply run
```bash
./lce_benchmark_model --graph=path_to_your_model.tflite
```

