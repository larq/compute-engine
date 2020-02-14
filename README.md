# Larq Compute Engine <img src="https://user-images.githubusercontent.com/13285808/74535800-84017780-4f2e-11ea-9169-52f5ac83d685.png" alt="larq logo" height="80px" align="right" />

[![Tests](https://github.com/larq/compute-engine/workflows/Tests/badge.svg)](https://github.com/larq/compute-engine/actions?workflow=Tests) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/larq-compute-engine.svg)](https://pypi.org/project/larq-compute-engine/) [![PyPI](https://img.shields.io/pypi/v/larq-compute-engine.svg)](https://pypi.org/project/larq-compute-engine/) [![PyPI - License](https://img.shields.io/pypi/l/larq-compute-engine.svg)](https://github.com/larq/compute-engine/blob/master/LICENSE) [![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/larq)

Larq Compute Engine (LCE) is a highly optimized inference engine for deploying
extremely quantized neural networks, such as
Binarized Neural Networks (BNNs). It currently supports various mobile platforms
and has been benchmarked on a Pixel 1 phone and a Raspberry Pi.
LCE provides a collection of hand-optimized [TensorFlow Lite](https://www.tensorflow.org/lite)
custom Ops for supported instruction sets, developed in inline assembly or in C++ 
using compiler intrinsics. LCE leverages optimization techniques
such as **tiling** to maximize the number of cache hits, **vectorization** to maximize 
the computational throughput, and **multi-threading parallelization** to take
advantage of multi-core modern desktop and mobile CPUs.

## Key Features
- **Effortless end-to-end integration** from training to deployment:

    - Tight integration of LCE with [Larq](https://larq.dev) and
      TensorFlow provides a smooth end-to-end training and deployment experience.

    - A collection of Larq pre-trained BNN models for common machine learning tasks
      is available in [Larq Zoo](https://larq.dev/models/)
      and can be used out-of-the-box with LCE.

    - LCE provides a custom [MLIR-based model converter](./docs/mlir_converter.md) which
      is fully compatible with TensorFlow Lite and performs additional
      network level optimizations for Larq models.

- **Lightning fast deployment** on a variety of mobile platforms:

    - LCE enables high performance, on-device machine learning inference by
      providing hand-optimized kernels and network level optimizations for BNN models.

    - LCE currently supports ARM64-based mobile platforms such as Android phones
      and Raspberry Pi boards.

    - Thread parallelism support in LCE is essential for modern mobile devices with
      multi-core CPUs.

## Performance
The table below presents **single-threaded** performance of Larq Compute Engine on
different versions of a novel BNN model called Quicknet (soon to be released in [larq-zoo](https://larq.dev/models/))
on the [Pixel 1 phone (2016)](https://support.google.com/pixelphone/answer/7158570?hl=en-GB)
and (Raspberry Pi 4 [BCM2711](https://www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2711/README.md)) board:

| Model          | Accuracy  | RPi 4 (BCM2711), ms | Pixel 1, ms   |
| -------------- | :-------: | :----------:        | :-----------: |
| Quicknet       | 57.6 %    | 60.5                | 28.1          |
| Quicknet-Large | 62.5 %    | 89.9                | 42.3          |

For reference, [dabnn](https://github.com/JDAI-CV/dabnn) (the other main BNN library) reports an inference time of 61.3 ms for [Bi-RealNet](https://larq.dev/api/larq_zoo/#birealnet) (56.4% accuracy) on the Pixel 1 phone,
while LCE achieves an inference time of 54.0 ms for BiRealNet on the same device.
They furthermore present a modified version, BiRealNet-Stem, which achieves the same accuracy of 56.4% in 43.2 ms.

The following table presents **multi-threaded** performance of Larq Compute Engine on
a Pixel 1 phone and a Raspberry Pi 4 board. These benchmarkes were performed with 4 cores:

| Model          | Accuracy  | RPi 4 (BCM2711), ms | Pixel 1, ms   |
| -------------- | :-------: | :----------:        | :-----------: |
| Quicknet       | 57.6 %    | 37.9                | 19.1          |
| Quicknet-Large | 62.5 %    | 55.8                | 28.0          |

Benchmarked on February 14th, 2020 with LCE custom
[TFLite Model Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
(see [here](./larq_compute_engine/tflite/benchmark))
and BNN models with randomized weights and inputs.

## Getting started
Follow these steps to deploy a BNN with LCE:

1. **Pick a Larq model**

    You can use [Larq](https://larq.dev) to build and train your own
    model or pick a pre-trained model from [Larq Zoo](https://larq.dev/models/).

1. **Convert the Larq model**

    LCE is built on top of TensorFlow Lite and uses TensorFlow Lite
    [FlatBuffer format](https://google.github.io/flatbuffers/)
    to convert and serialize Larq models for inference.
    We provide a [LCE Converter](./docs/mlir_converter.md) with additional
    optimization passes to increase the speed of execution of Larq models
    on supported target platforms.

1. **Build LCE**

    The LCE documentation provides the build instructions for [Android](./docs/quickstart_android.md)
    and [ARM64-based boards](./docs/build_arm.md) such as Raspberry Pi.
    Please follow the provided instructions to create a native LCE build
    or cross-compile for one of the supported targets.


1. **Run inference**

    LCE uses the [TensorFlow Lite Interpreter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/inference.md) 
    to perform an inference. In addition to the already available built-in
    TensorFlow Lite Ops, optimized LCE Ops are registered to the interpreter
    to execute the Larq specific subgraphs of the model. An example to create
    and build LCE compatible TensorFlow Lite interpreter in user's applications
    is provided [here](./docs/inference.md).

## Next steps
- Explore [Larq pre-trained models](https://larq.dev/models/).
- Learn how to [build](https://larq.dev/guides/bnn-architecture/) and
  [train](https://larq.dev/guides/bnn-optimization/) BNNs for your own
  application with Larq.
- Try our [example programs](./examples/).
