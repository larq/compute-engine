# Get started with Larq Compute Engine
The [Larq](https://larq.dev/) toolchain provides all the tools you need to
[train](1.-Choose-a-Larq-Model), [convert](2.-Convert-a-Larq-Model) and
perform [inference](3.-Run-inference-with-LCE) with neural networks with
extremely low-precision weights and activations,
such as Binarized Neural Networks (BNNs). In this guide, we give you a general
overview of Larq toolchain and provide links to in-depth documentation of
each of the Larq toolchain components.

## 1. Choose a Larq model
Depending on your application, you need to choose an appropriate neural network.
You can either train your own BNN model with [Larq](https://larq.dev/)
or use one of the Larq pretrained models in [Larq Zoo](https://larq.dev/models/).

Larq is [open-source](https://github.com/larq/larq) and provides a comperehensive
collection of [documenatation](https://github.com/larq/larq/tree/master/docs),
[tutorials](https://larq.dev/guides/key-concepts/) and
[examples](https://larq.dev/examples/mnist/) to train BNN models.
Additionaly, Larq Zoo provides a [collection](https://larq.dev/models/)
of reference implementation of BNNs with pretained weights.

## 2. Convert a Larq model
To execute BNN models with Larq Compute Engine (LCE), you must convert a Larq
model to a LCE-compatible TenforFlow Lite FlatBuffer. LCE provids a
[MLIR](https://www.tensorflow.org/mlir)-based [converter](./mlir_converter.md)
which is built on top of the
[TensorFlow Lite converter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/get_started.md#2-convert-the-model)
and performs additional network level optimizations for Larq BNN models.
LCE MLIR Converter provides C++ as well as Python API to convert the Larq models.
To learn more about LCE MLIR converter see [here](./mlir_converter.md).

## 3. Run inference with LCE
To perform inference of a Larq converted model, LCE relies on TensorFlow Lite
interpreter which schedules the execution of the operations defined in the model.
LCE provies highly-optimized custom Ops which can be registerd by the interprter
to be used instead of built-in TensorFlow Lite Ops for each applicable subgraph
of the model.
To create you own custom LCE inference binary, explore the [LCE inference guide](./inference.md).
See also the [Android quickstart](quickstart_android.md) and [ARM-boards build guide](build_arm.md) 
to port your inference binary to supported LCE platforms.
