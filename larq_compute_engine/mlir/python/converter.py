from packaging import version
import warnings
from typing import Optional, Tuple
import tensorflow as tf

from larq_compute_engine.mlir._graphdef_tfl_flatbuffer import (
    convert_graphdef_to_tflite_flatbuffer,
)
from larq_compute_engine.mlir.python.util import modify_integer_quantized_model_io_type

from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.lite.python.util import get_tensor_name
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


def _contains_training_quant_op(graph_def):
    """Checks if the graph contains any training-time quantization ops."""
    training_quant_ops = {
        "FakeQuantWithMinMaxVars",
        "FakeQuantWithMinMaxVarsPerChannel",
        "FakeQuantWithMinMaxArgs",
        "FakeQuantWithMinMaxArgsPerChannel",
        "QuantizeAndDequantizeV2",
        "QuantizeAndDequantizeV3",
    }

    for node_def in graph_def.node:
        if node_def.op in training_quant_ops:
            return True
    for function in graph_def.library.function:
        for node_def in function.node_def:
            if node_def.op in training_quant_ops:
                return True
    return False


def convert_keras_model(
    model: tf.keras.Model,
    *,  # Require remaining arguments to be keyword-only.
    inference_input_type: tf.DType = tf.float32,
    inference_output_type: tf.DType = tf.float32,
    target: str = "arm",
    experimental_default_int8_range: Optional[Tuple[float, float]] = None,
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
        inference_input_type: Data type of the input layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        inference_output_type: Data type of the output layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        target: Target hardware platform. Must be "arm" or "xcore".
        experimental_default_int8_range: Tuple of integers representing `(min, max)`
            range values for all arrays without a specified range. Intended for
            experimenting with quantization via "dummy quantization". (default None)
        experimental_enable_bitpacked_activations: Enable an experimental
            converter optimisation that attempts to reduce intermediate
            activation memory usage by bitpacking the activation tensor between
            consecutive binary convolutions where possible.

    # Returns
        The converted data in serialized format.
    """
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            f"Expected `model` argument to be a `tf.keras.Model` instance, got `{model}`."
        )
    if inference_input_type not in (tf.float32, tf.int8):
        raise ValueError(
            "Expected `inference_input_type` to be either `tf.float32` or `tf.int8`, "
            f"got {inference_input_type}."
        )
    if inference_output_type not in (tf.float32, tf.int8):
        raise ValueError(
            "Expected `inference_output_type` to be either `tf.float32` or `tf.int8`, "
            f"got {inference_output_type}."
        )
    if target not in ("arm", "xcore"):
        raise ValueError(f'Expected `target` to be "arm" or "xcore", but got {target}.')

    if not tf.executing_eagerly():
        raise RuntimeError(
            "Graph mode is not supported. Please enable eager execution using "
            "tf.enable_eager_execution() when using TensorFlow 1.x"
        )
    if experimental_default_int8_range:
        warnings.warn(
            "Using `experimental_default_int8_range` as fallback quantization stats. "
            "This should only be used for latency tests."
        )
    if hasattr(model, "dtype_policy") and model.dtype_policy.name != "float32":
        raise RuntimeError(
            "Mixed precision float16 models are not supported by the TFLite converter, "
            "please convert them to float32 first. See also: "
            "https://github.com/tensorflow/tensorflow/issues/46380"
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
    should_quantize = (
        _contains_training_quant_op(graph_def)
        or experimental_default_int8_range is not None
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

    tflite_buffer = convert_graphdef_to_tflite_flatbuffer(
        graph_def.SerializeToString(),
        [get_tensor_name(tensor) for tensor in input_tensors],
        [DataType.Name(tensor.dtype.as_datatype_enum) for tensor in input_tensors],
        [tensor.shape.as_list() for tensor in input_tensors],
        [get_tensor_name(tensor) for tensor in output_tensors],
        should_quantize,
        target,
        experimental_default_int8_range,
        experimental_enable_bitpacked_activations,
    )
    if should_quantize and (
        inference_input_type != tf.float32 or inference_output_type != tf.float32
    ):
        tflite_buffer = modify_integer_quantized_model_io_type(
            tflite_buffer,
            inference_input_type=inference_input_type,
            inference_output_type=inference_output_type,
        )

    return tflite_buffer
