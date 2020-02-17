# Larq Compute Engine Converter

The LCE converter allows you to convert a Keras model built with `larq` to an LCE-compatible TensorFlow Lite FlatBuffer format for inference.

## Installation

Before installing the LCE converter, please install:

- [Python](https://www.python.org/) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.14`, `1.15`, `2.0.0`, or `2.1.0` (recommended):
  ```shell
  pip install tensorflow  # or tensorflow-gpu
  ```

You can install the LCE converter with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq-compute-engine
```

## Converting a Larq model

Convert your model and write the file to disk:
```python
import larq_compute_engine as lce

model = ... # Your custom Keras model or one from larq_zoo
with open("/tmp/my_model.tflite", "wb") as f:
    f.write(lce.convert_keras_model(model))
```
