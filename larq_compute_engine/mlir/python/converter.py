from packaging import version
from typing import Union, Callable, Iterator
import numpy as np
import tensorflow as tf

from larq_compute_engine.mlir._graphdef_tfl_flatbuffer import (
    convert_graphdef_to_tflite_flatbuffer,
)
from larq_compute_engine.mlir.quantization._calibration_wrapper import Calibrator

from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.lite.python.util import (
    get_grappler_config,
    get_tensor_name,
    run_graph_optimizations,
)
from tensorflow.python.eager import def_function
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from tensorflow.python.keras.saving import saving_utils


def concrete_function_from_keras_model(model):
    input_signature = None
    if version.parse(tf.__version__) >= version.parse("2.1"):
        # If the model's call is not a `tf.function`, then we need to first get its
        # input signature from `model_input_signature` method. We can't directly
        # call `trace_model_call` because otherwise the batch dimension is set
        # to None.
        # Once we have better support for dynamic shapes, we can remove this.
        if not isinstance(model.call, def_function.Function):
            # Pass `keep_original_batch_size=True` will ensure that we get an input
            # signature including the batch dimension specified by the user.
            input_signature = saving_utils.model_input_signature(
                model, keep_original_batch_size=True
            )

    func = saving_utils.trace_model_call(model, input_signature)
    return func.get_concrete_function()


def calibrate_and_quantize(
    flatbuffer_model: bytes,
    dataset_gen: Callable[[], Iterator],
    input_type: Union[type, str] = np.float32,
    output_type: Union[type, str] = np.float32,
) -> bytes:
    """Use post training quantization to convert floating point operations to integer.

    !!! example
        ```python
        def dataset_gen():
            for _ in range(3):
                yield [np.random.normal(size=(1, 224, 224, 3)).astype(np.float32)]

        tflite_model = convert_keras_model(model)
        tflite_model = calibrate_and_quantize(tflite_model, dataset_gen)
        with open("/tmp/my_model.tflite", "wb") as f:
            f.write(tflite_model)
        ```

    # Arguments
    flatbuffer_model: A TFLite flatbuffer model.
    dataset_gen: An input generator that can be used to generate input samples for the
        model. This must be a callable object that returns an object that supports the
        `iter()` protocol (e.g. a generator function). The elements generated must have
        same type and shape as inputs to the model.
    input_type: A `str` or `type` representing the type of the input arrays.
    output_type: A `str` or `type` representing the type of the model output.

    # Returns
    The quantized model in serialized format.
    """
    calibrator = Calibrator(flatbuffer_model)
    calibrator.Prepare()
    # Run the images through the model
    for calibration_sample in dataset_gen():
        calibrator.FeedTensor(calibration_sample)
    return calibrator.QuantizeModel(
        np.dtype(input_type).num, np.dtype(output_type).num,
    )


def convert_keras_model(
    model: tf.keras.Model,
    *,  # Require remaining arguments to be keyword-only.
    experimental_enable_bitpacked_activations: bool = False,
) -> bytes:
    """Converts a Keras model to TFLite flatbuffer.

    !!! example
        ```python
        tflite_model = convert_keras_model(model)
        with open("/tmp/my_model.tflite", "wb") as f:
            f.write(tflite_model)
        ```

    # Arguments
        model: The model to convert.
        experimental_enable_bitpacked_activations: Enable an experimental
            converter optimisation that attempts to reduce intermediate
            activation memory usage by bitpacking the activation tensor between
            consecutive binary convolutions where possible.

    # Returns
        The converted model in serialized format.
    """
    if not tf.executing_eagerly():
        raise RuntimeError(
            "Graph mode is not supported. Please enable eager execution using "
            "tf.enable_eager_execution() when using TensorFlow 1.x"
        )
    func = concrete_function_from_keras_model(model)
    if version.parse(tf.__version__) >= version.parse("1.15"):
        frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
    else:
        frozen_func = convert_variables_to_constants_v2(func)
    input_tensors = [
        tensor for tensor in frozen_func.inputs if tensor.dtype != tf.dtypes.resource
    ]
    output_tensors = frozen_func.outputs

    graph_def = frozen_func.graph.as_graph_def()
    # Run a constant folding using grappler since we currently don't implement
    # folding for LCE custom ops
    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold"]),
        graph=frozen_func.graph,
    )

    # Checks dimensions in input tensor.
    for tensor in input_tensors:
        # Note that shape_list might be empty for scalar shapes.
        shape_list = tensor.shape.as_list()
        if None in shape_list[1:]:
            raise ValueError(
                "None is only supported in the 1st dimension. Tensor '{0}' has "
                "invalid shape '{1}'.".format(get_tensor_name(tensor), shape_list)
            )
        elif shape_list and shape_list[0] is None:
            # Set the batch size to 1 if undefined.
            shape = tensor.shape.as_list()
            shape[0] = 1
            tensor.set_shape(shape)

    return convert_graphdef_to_tflite_flatbuffer(
        graph_def.SerializeToString(),
        [get_tensor_name(tensor) for tensor in input_tensors],
        [DataType.Name(tensor.dtype.as_datatype_enum) for tensor in input_tensors],
        [tensor.shape.as_list() for tensor in input_tensors],
        [get_tensor_name(tensor) for tensor in output_tensors],
        experimental_enable_bitpacked_activations,
    )
