# Larq Compute Engine <img src="https://user-images.githubusercontent.com/13285808/74535800-84017780-4f2e-11ea-9169-52f5ac83d685.png" alt="larq logo" height="80px" align="right" />

[![Tests](https://github.com/larq/compute-engine/workflows/Tests/badge.svg)](https://github.com/larq/compute-engine/actions?workflow=Tests) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/larq-compute-engine.svg)](https://pypi.org/project/larq-compute-engine/) [![PyPI](https://img.shields.io/pypi/v/larq-compute-engine.svg)](https://pypi.org/project/larq-compute-engine/) [![PyPI - License](https://img.shields.io/pypi/l/larq-compute-engine.svg)](https://github.com/larq/compute-engine/blob/main/LICENSE)

Larq Compute Engine (LCE) is a highly optimized inference engine for deploying
extremely quantized neural networks, such as
Binarized Neural Networks (BNNs). It currently supports various mobile platforms
and has been benchmarked on a Pixel 1 phone and a Raspberry Pi.
LCE provides a collection of hand-optimized [TensorFlow Lite](https://www.tensorflow.org/lite)
custom operators for supported instruction sets, developed in inline assembly or in C++
using compiler intrinsics. LCE leverages optimization techniques
such as **tiling** to maximize the number of cache hits, **vectorization** to maximize
the computational throughput, and **multi-threading parallelization** to take
advantage of multi-core modern desktop and mobile CPUs.

*Larq Compute Engine is part of a family of libraries for BNN development; you can also check out [Larq](https://github.com/larq/larq) for building and training BNNs and [Larq Zoo](https://github.com/larq/zoo) for pre-trained models.*

## Key Features

- **Effortless end-to-end integration** from training to deployment:

    - Tight integration of LCE with [Larq](https://larq.dev) and
      TensorFlow provides a smooth end-to-end training and deployment experience.

    - A collection of Larq pre-trained BNN models for common machine learning tasks
      is available in [Larq Zoo](https://docs.larq.dev/zoo/)
      and can be used out-of-the-box with LCE.

    - LCE provides a custom [MLIR-based model converter](https://docs.larq.dev/compute-engine/api/converter) which
      is fully compatible with TensorFlow Lite and performs additional
      network level optimizations for Larq models.

- **Lightning fast deployment** on a variety of mobile platforms:

    - LCE enables high performance, on-device machine learning inference by
      providing hand-optimized kernels and network level optimizations for BNN models.

    - LCE currently supports 64-bit ARM-based mobile platforms such as Android phones
      and Raspberry Pi boards.

    - Thread parallelism support in LCE is essential for modern mobile devices with
      multi-core CPUs.

## Performance

The table below presents **single-threaded** performance of Larq Compute Engine on
different versions of a novel BNN model called QuickNet (trained on ImageNet dataset, released on [Larq Zoo](https://docs.larq.dev/zoo/))
on a Raspberry Pi 4 Model B at 1.5GHz ([BCM2711](https://www.raspberrypi.com/documentation/computers/processors.html#bcm2711)) board, a [Pixel 1 Android phone (2016)](https://support.google.com/pixelphone/answer/7158570?hl=en-GB), and a [Mac Mini with M1 ARM CPU](https://www.apple.com/uk/mac-mini/):

| Model                                                              | Top-1 Accuracy | RPi 4B 1.5GHz, 1 thread (ms) | Pixel 1, 1 thread (ms) | Mac Mini M1, 1 thread (ms) |
|--------------------------------------------------------------------|----------------|------------------------------|------------------------|----------------------------|
| [QuickNetSmall](https://docs.larq.dev/zoo/api/sota/#quicknetsmall) | 59.4%          | 27.7                         | 16.8                   | 4.0                        |
| [QuickNet](https://docs.larq.dev/zoo/api/sota/#quicknet)           | 63.3%          | 45.0                         | 25.5                   | 5.8                        |
| [QuickNetLarge](https://docs.larq.dev/zoo/api/sota/#quicknetlarge) | 66.9%          | 77.0                         | 44.2                   | 9.9                        |

For reference, [dabnn](https://github.com/JDAI-CV/dabnn) (the other main BNN library) reports an inference time of 61.3 ms for [Bi-RealNet](https://docs.larq.dev/zoo/api/literature/#birealnet) (56.4% accuracy) on the Pixel 1 phone,
while LCE achieves an inference time of 41.6 ms for Bi-RealNet on the same device.
They furthermore present a modified version, BiRealNet-Stem, which achieves the same accuracy of 56.4% in 43.2 ms.

The following table presents **multi-threaded** performance of Larq Compute Engine on
a Pixel 1 phone and a Raspberry Pi 4 Model B at 1.5GHz ([BCM2711](https://www.raspberrypi.com/documentation/computers/processors.html#bcm2711))
board:

| Model                                                              | Top-1 Accuracy | RPi 4B 1.5GHz, 4 threads (ms) | Pixel 1, 4 threads (ms) | Mac Mini M1, 4 threads (ms) |
|--------------------------------------------------------------------|----------------|-------------------------------|-------------------------|-----------------------------|
| [QuickNetSmall](https://docs.larq.dev/zoo/api/sota/#quicknetsmall) | 59.4%          | 12.1                          | 8.9                     | 1.8                         |
| [QuickNet](https://docs.larq.dev/zoo/api/sota/#quicknet)           | 63.3%          | 20.8                          | 12.6                    | 2.5                         |
| [QuickNetLarge](https://docs.larq.dev/zoo/api/sota/#quicknetlarge) | 66.9%          | 31.7                          | 22.8                    | 3.9                         |

Benchmarked on 2021-06-11 (Pixel 1), 2021-06-13 (Mac Mini M1), and 2022-04-20 (RPi 4B) with LCE custom
[TFLite Model Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
(see [here](https://github.com/larq/compute-engine/tree/main/larq_compute_engine/tflite/benchmark))
with XNNPack enabled
and BNN models with randomized inputs.

## Getting started

Follow these steps to deploy a BNN with LCE:

1. **Pick a Larq model**

    You can use [Larq](https://larq.dev) to build and train your own model or pick a pre-trained model from [Larq Zoo](https://docs.larq.dev/zoo/).

2. **Convert the Larq model**

    LCE is built on top of TensorFlow Lite and uses the TensorFlow Lite [FlatBuffer format](https://google.github.io/flatbuffers/) to convert and serialize Larq models for inference. We provide an [LCE Converter](https://docs.larq.dev/compute-engine/api/converter) with additional optimization passes to increase the speed of execution of Larq models on supported target platforms.

3. **Build LCE**

    The LCE documentation provides the build instructions for [Android](https://docs.larq.dev/compute-engine/quickstart_android) and [64-bit ARM-based boards](https://docs.larq.dev/compute-engine/build/arm) such as Raspberry Pi. Please follow the provided instructions to create a native LCE build or cross-compile for one of the supported targets.

4. **Run inference**

    LCE uses the [TensorFlow Lite Interpreter](https://www.tensorflow.org/lite/guide/inference) to perform an inference. In addition to the already available built-in TensorFlow Lite operators, optimized LCE operators are registered to the interpreter to execute the Larq specific subgraphs of the model. An example to create and build an LCE compatible TensorFlow Lite interpreter for your own applications is provided [here](https://docs.larq.dev/compute-engine/inference).

## Next steps

- Explore [Larq pre-trained models](https://docs.larq.dev/zoo/).
- Learn how to [build](https://docs.larq.dev/larq/guides/bnn-architecture/) and
  [train](https://docs.larq.dev/larq/guides/bnn-optimization/) BNNs for your own
  application with Larq.
- If you're a mobile developer, visit [Android quickstart](https://docs.larq.dev/compute-engine/quickstart_android).
- See our build instructions for Raspberry Pi and 64-bit ARM-based boards [here](https://docs.larq.dev/compute-engine/build/arm).
- Try our [example programs](https://github.com/larq/compute-engine/tree/main/examples).

## About

Larq Compute Engine is being developed by a team of deep learning researchers and engineers at Plumerai to help accelerate both our own research and the general adoption of Binarized Neural Networks.
