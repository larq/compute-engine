import os
from packaging import version
import warnings
from typing import Optional, Tuple, Union
import tensorflow as tf
import tempfile

from larq_compute_engine.mlir._tf_tfl_flatbuffer import (
    convert_graphdef_to_tflite_flatbuffer,
    convert_saved_model_to_tflite_flatbuffer,
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


def _validate_options(
    *,
    inference_input_type=None,
    inference_output_type=None,
    target=None,
    experimental_default_int8_range=None,
):
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


def convert_saved_model(
    saved_model_dir: Union[str, os.PathLike],
    *,  # Require remaining arguments to be keyword-only.
    inference_input_type: tf.DType = tf.float32,
    inference_output_type: tf.DType = tf.float32,
    target: str = "arm",
    experimental_default_int8_range: Optional[Tuple[float, float]] = None,
) -> bytes:
    """Converts a SavedModel to TFLite flatbuffer.

    !!! example
        ```python
        tflite_model = convert_saved_model(saved_model_dir)
        with open("/tmp/my_model.tflite", "wb") as f:
            f.write(tflite_model)
        ```

    # Arguments
        saved_model_dir: SavedModel directory to convert.
        inference_input_type: Data type of the input layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        inference_output_type: Data type of the output layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        target: Target hardware platform. Defaults to "arm", must be either "arm"
            or "xcore".
        experimental_default_int8_range: Tuple of integers representing `(min, max)`
            range values for all arrays without a specified range. Intended for
            experimenting with quantization via "dummy quantization". (default None)

    # Returns
        The converted data in serialized format.
    """
    if version.parse(tf.__version__) < version.parse("2.2"):
        raise RuntimeError(
            "TensorFlow 2.2 or newer is required for saved model conversion."
        )
    _validate_options(
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        target=target,
        experimental_default_int8_range=experimental_default_int8_range,
    )

    saved_model_dir = str(saved_model_dir)
    saved_model_tags = [tf.saved_model.SERVING]
    saved_model_exported_names = [tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    from tensorflow.python.saved_model import loader_impl

    saved_model_pb, _ = loader_impl.parse_saved_model_with_debug_info(saved_model_dir)

    saved_model_version = saved_model_pb.saved_model_schema_version
    if saved_model_version not in (1, 2):
        raise ValueError(
            f"SavedModel file format({saved_model_version}) is not supported"
        )

    tflite_buffer = convert_saved_model_to_tflite_flatbuffer(
        saved_model_dir,
        saved_model_tags,
        saved_model_exported_names,
        saved_model_version,
        target,
        experimental_default_int8_range,
    )

    if inference_input_type != tf.float32 or inference_output_type != tf.float32:
        tflite_buffer = modify_integer_quantized_model_io_type(
            tflite_buffer,
            inference_input_type=inference_input_type,
            inference_output_type=inference_output_type,
        )

    return tflite_buffer


def convert_keras_model(
    model: tf.keras.Model,
    *,  # Require remaining arguments to be keyword-only.
    inference_input_type: tf.DType = tf.float32,
    inference_output_type: tf.DType = tf.float32,
    target: str = "arm",
    experimental_default_int8_range: Optional[Tuple[float, float]] = None,
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
        target: Target hardware platform. Defaults to "arm", must be either "arm"
            or "xcore".
        experimental_default_int8_range: Tuple of integers representing `(min, max)`
            range values for all arrays without a specified range. Intended for
            experimenting with quantization via "dummy quantization". (default None)

    # Returns
        The converted data in serialized format.
    """
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            f"Expected `model` argument to be a `tf.keras.Model` instance, got `{model}`."
        )
    if hasattr(model, "dtype_policy") and model.dtype_policy.name != "float32":
        raise ValueError(
            "Mixed precision float16 models are not supported by the TFLite converter, "
            "please convert them to float32 first. See also: "
            "https://github.com/tensorflow/tensorflow/issues/46380"
        )
    _validate_options(
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        target=target,
        experimental_default_int8_range=experimental_default_int8_range,
    )

    # First attempt conversion as saved model
    try:
        with tempfile.TemporaryDirectory() as saved_model_dir:
            model.save(saved_model_dir, save_format="tf")

            return convert_saved_model(
                saved_model_dir,
                inference_input_type=inference_input_type,
                inference_output_type=inference_output_type,
                experimental_default_int8_range=experimental_default_int8_range,
                target=target,
            )
    except Exception:
        warnings.warn(
            "Saved-model conversion failed, falling back to graphdef-based conversion."
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
