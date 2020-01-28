"""TF lite converter for larq models."""
import tensorflow as tf
import numpy as np
import larq as lq
import subprocess
from larq_compute_engine import bsign, bconv2d64
from larq_compute_engine.python.utils import tf_2_or_newer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func

get_custom_objects()["bsign"] = bsign


quantizer_replacements = {
    "SteSign": bsign,
    "ste_sign": bsign,
    "approx_sign": bsign,
    "MagnitudeAwareSign": None,
    "magnitude_aware_sign": None,
    "swish_sign": bsign,
    "SwishSign": bsign,
    "SteTern": None,
    "ste_tern": None,
    "SteHeaviside": None,
    "ste_heaviside": None,
    "DoReFaQuantizer": None,
    "dorefa_quantizer": None,
}


def create_bconv_layer(
    weights, strides, padding, transpose=True, fused_multiply=None, fused_add=None
):
    """
    Creates a binary convolution layer for tflite.

    If `transpose` is True, transposes from HWIO to OHWI

    When `fused_multiply` is not `None`, it should be a 1D array of size equal to the filter out-channel dimension.
    In this case, a multiplication op is inserted *after* the convolution.
    This multiplication op will be merged into the batchnorm op by the converter.
    This has two purposes:
    - Implement the multiplication for the back-transformation from {0,1} to {-1,1}
    - Implement the multiplication for magnitude_aware_sign in BiRealNet
    """
    strides = [1, strides[0], strides[1], 1]
    padding = padding.upper()

    # Here the weights are still HWIO
    dotproduct_size = weights.shape[0] * weights.shape[1] * weights.shape[2]

    filter_format = "HWIO"
    if transpose:
        # Transpose: change from HWIO to OHWI
        weights = np.moveaxis(weights, 3, 0)
        filter_format = "OHWI"
        weights = np.sign(np.sign(weights) + 0.5)

    out_channels = weights.shape[0]

    if fused_multiply is None:
        fused_multiply = np.full(shape=(out_channels), fill_value=1)
    elif len(fused_multiply.shape) != 1 or fused_multiply.shape[0] != out_channels:
        raise Exception(
            f"ERROR: Argument fused_multiply should have shape ({weights.shape[0]}) but has shape {fused_multiply.shape}"
        )

    if fused_add is None:
        fused_add = np.full(shape=(out_channels), fill_value=0)
    elif len(fused_add.shape) != 1 or fused_add.shape[0] != out_channels:
        raise Exception(
            f"ERROR: Argument fused_add should have shape ({weights.shape[0]}) but has shape {fused_add.shape}"
        )

    # The bconv will do the following:
    # output = fused_add[channel] + fused_multiply[channel] * popcount
    # We use this to implement two things:
    # - `y1 = n - 2 * popcount`     (the backtransformation to -1,+1 space)
    # - `y2 = a + b * y1`           (optional fused batchnorm)
    # Together they become
    # `y = (a + b*n) + (-2b) * popcount
    fused_add = fused_add + dotproduct_size * fused_multiply
    fused_multiply = -2 * fused_multiply

    def bconv_op(x):
        y = bconv2d64(
            x,
            weights,
            fused_multiply,
            fused_add,
            strides,
            padding,
            data_format="NHWC",
            filter_format=filter_format,
        )
        return y

    return bconv_op


def replace_layers(model, replacement_dict):
    """
    This function is adapted from
    https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model

    Note: it currently fails on complicated networks such as two networks in parallel, i.e. two input tensors, run separate models on them and have two output tensors, but the whole thing viewed as one network.

    However, we will probably switch to another conversion method once we understand grappler and other tensorflow parts, so for now this method is fine because it works on all Larq models.
    """
    # Auxiliary dictionary to describe the network graph
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict["input_layers_of"]:
                network_dict["input_layers_of"].update({layer_name: [layer.name]})
            else:
                network_dict["input_layers_of"][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict["new_output_tensor_of"].update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        if not layer.name in network_dict["input_layers_of"]:
            print(f"ERROR: {layer.name} not in input_layers_of")
            return None

        # Determine input tensors
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if layer.name in replacement_dict:
            x = layer_input

            new_layer = replacement_dict[layer.name]
            new_layer.name = "{}_new".format(layer.name)

            x = new_layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict["new_output_tensor_of"].update({layer.name: x})

    return tf.keras.Model(inputs=model.inputs, outputs=x)


class ModelConverter:
    """Converter to create TF lite models from Larq Keras models

    This converter will convert the input quantizers to their tflite counterpart.
    It will remove the kernel quantizers and instead only store the signs of the latent weights.

    By default the converter will try out different options and stop as soon as one option succeeded.
    By setting the option `converter.do_all_methods = True` one can enforce that all options will be executed and saved as `filename_XXX.tflite`

    By setting the option `converter.quantize = True`, the non-binary parts of the network will be 8-bit quantized.
    If quantization is used, then one has to set `converter.representative_dataset_gen`, as required for post-training quantization in Tensorflow. See [this example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_integer_quant.ipynb).

    # Arguments
    model: The Keras model to convert.

    !!! example
        ```python
        from larq_zoo import BiRealNet
        model = BiRealNet(weights="imagenet")
        conv = ModelConverter(model)
        tflite_model = conv.convert()
        # Or directly save it to a file:
        conv.convert("birealnet.tflite")
        ```
    """

    def __init__(self, model):
        self.model = model
        self.do_all_methods = False
        self.has_custom_ops = False
        self.quantize = False
        self.representative_dataset_gen = None

    def convert(self, filename=None):
        """Convert and return the tflite model.

        Optionally save the model to a file.

        # Arguments
        filename: If `None`, then returns the tflite model object. If its a string then it saves the model to that filename.
        """
        if self.do_all_methods and filename is None:
            print("If `do_all_methods` is enabled then filename can not be None")

        if not self.fix_quantizers():
            print("Model contains unsupported quantizers. No conversion will be done.")
            return None

        # Methods should be listed in order of most likely to succeed
        # because normally the converter stops after the first succesful conversion
        if self.quantize:
            methods = [
                (
                    "compatv1_train",
                    lambda: self.convert_compatv1(
                        train_quantize=True, extra_shape_inference=False
                    ),
                ),
                (
                    "compatv1_train_withshapes",
                    lambda: self.convert_compatv1(
                        train_quantize=True, extra_shape_inference=True
                    ),
                ),
                (
                    "mlir_post",
                    lambda: self.convert_fromkeras(
                        new_converter=True, post_quantize=True
                    ),
                ),
                (
                    "toco_post",
                    lambda: self.convert_fromkeras(
                        new_converter=False, post_quantize=True
                    ),
                ),
                ("compatv1_post", lambda: self.convert_compatv1(post_quantize=True),),
            ]
        else:
            methods = [
                (
                    "mlir",
                    lambda: self.convert_fromkeras(
                        new_converter=True, post_quantize=False
                    ),
                ),
                (
                    "toco",
                    lambda: self.convert_fromkeras(
                        new_converter=False, post_quantize=False
                    ),
                ),
                (
                    "compatv1",
                    lambda: self.convert_compatv1(extra_shape_inference=False),
                ),
                (
                    "compatv1_withshapes",
                    lambda: self.convert_compatv1(extra_shape_inference=True),
                ),
            ]

        tflite_models = []
        result_log = []
        result_summary = []

        for name, conv_func in methods:
            print(f"Running conversion method: {name}")
            try:
                tflite_model = conv_func()
                result_summary.append(f"{name} method: success")
                tflite_models.append((name, tflite_model))
                if not self.do_all_methods:
                    break
            except Exception as e:
                result_log.append(f"{name} method log:\n{str(e)}")
                result_summary.append(f"{name} method: failed")

        print("\n----------------\nConversion logs:")
        for log in result_log:
            print("----------------")
            print(log)

        print("----------------\nConversion summary:")
        for log in result_summary:
            print(log)

        first_model = tflite_models[0][1] if len(tflite_models) >= 1 else None

        if self.do_all_methods:
            for name, tflite_model in tflite_models:
                fn = f"{filename}_{name}.tflite"
                print(f"Saving tf lite model as {fn}")
                open(fn, "wb").write(tflite_model)
        else:
            if first_model and filename is not None:
                print(f"Saving tf lite model as {filename}")
                open(filename, "wb").write(first_model)

        return first_model

    def fix_quantizers(self):
        result = True

        replacement_dict = {}
        for l in self.model.layers:
            supported_input_quantizer = False
            supported_kernel_quantizer = False
            mul_weights = None

            input_quantizer = None
            try:
                input_quantizer = l.input_quantizer
            except AttributeError:
                pass
            if input_quantizer is not None:
                name = lq.quantizers.serialize(input_quantizer)
                if isinstance(name, dict):
                    name = name["class_name"]
                if not isinstance(name, str) or name not in quantizer_replacements:
                    print(f"ERROR: Input quantizer {name} unknown.")
                    result = False
                elif quantizer_replacements[name] is None:
                    print(f"ERROR: Input quantizer {name} not yet supported.")
                    result = False
                else:
                    l.input_quantizer = quantizer_replacements[name]
                    supported_input_quantizer = True

            kernel_quantizer = None
            try:
                kernel_quantizer = l.kernel_quantizer
            except AttributeError:
                pass
            if kernel_quantizer is None:
                # When its trained with Bop then it doesn't have kernel quantizers
                # So for QuantConv2D just assume its a binary kernel
                if isinstance(l, lq.layers.QuantConv2D):
                    supported_kernel_quantizer = True
            else:
                name = lq.quantizers.serialize(kernel_quantizer)
                if isinstance(name, dict):
                    name = name["class_name"]
                if not isinstance(name, str) or name not in quantizer_replacements:
                    print(f"ERROR: Kernel quantizer {name} unknown.")
                    result = False
                elif name == "magnitude_aware_sign":
                    w = l.get_weights()[0]
                    absw = np.abs(w)
                    means = np.mean(absw, axis=tuple(range(len(w.shape) - 1)))
                    mul_weights = means
                    supported_kernel_quantizer = True
                    # l.set_weights([means * np.sign(np.sign(w) + 0.5)])
                elif quantizer_replacements[name] is None:
                    print(f"ERROR: Kernel quantizer {name} not yet supported.")
                    result = False
                else:
                    supported_kernel_quantizer = True

            if supported_input_quantizer and supported_kernel_quantizer:
                l.kernel_quantizer = None

                # TODO: Our QuantConv2D layers usually don't have a bias because it is already in the batchnorm.
                # If they do, however, then we have to check whether its [0] or [1] here based on shape
                w = l.get_weights()[0]

                if isinstance(l, lq.layers.QuantConv2D):
                    if len(w.shape) != 4:
                        print(
                            f"ERROR: Weights of layer {l.name} have shape {w.shape} which does not have rank 4."
                        )
                        result = False
                    else:
                        # Create a new layer with those weights
                        # TODO: Detect if there is a batchnorm and put that into
                        # fused_multiply, fused_add
                        bconvlayer = create_bconv_layer(
                            w,
                            l.strides,
                            l.padding,
                            transpose=True,
                            fused_multiply=mul_weights,
                            fused_add=None,
                        )
                        replacement_dict[l.name] = bconvlayer
                else:
                    binary_weights = np.sign(np.sign(w) + 0.5)
                    l.set_weights([binary_weights])

        if result and replacement_dict:
            self.has_custom_ops = True
            new_model = replace_layers(self.model, replacement_dict)
            if new_model is None:
                return False
            else:
                self.model = new_model

        return result

    def convert_fromkeras(self, new_converter=False, post_quantize=False):
        """Conversion using the v2 converter.
        """
        if tf_2_or_newer():
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        else:
            keras_file = "/tmp/modelconverter_temporary.h5"
            tf.keras.models.save_model(self.model, keras_file)
            converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
        converter.allow_custom_ops = True

        if post_quantize:
            if self.has_custom_ops:
                raise Exception(
                    "Post-training quantization is not yet supported for custom ops."
                )

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_gen
            # The lines below enforce full int8 (error on unsupported op)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            # The `inference_xxx_type` parameters are ignored in tf2 and reset to float

        if new_converter:
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            # For now, we print a note to the user on how to add our custom op to the MLIR converter
            toco_path = (
                subprocess.check_output("which toco_from_protos", shell=True)
                .strip()
                .decode("ascii")
            )
            print(
                f"Note: to use the MLIR converter, please add the line `import larq_compute_engine as lqce` to `{toco_path}`"
            )
        return converter.convert()

    def convert_compatv1(
        self,
        train_quantize=False,
        post_quantize=False,
        extra_shape_inference=False,
        new_converter=False,
    ):
        """Conversion using the compat.v1 interface of the converter.
        """
        # First convert the graph to a 'frozen graph def' so that we can edit the attributes

        # This frozen graph part is taken from the compat.v1 converter in `lite/python/lite.py`
        tf.keras.backend.set_learning_phase(False)
        function = _saving_utils.trace_model_call(self.model)
        concrete_func = function.get_concrete_function()
        frozen_func = _convert_to_constants.convert_variables_to_constants_v2(
            concrete_func, lower_control_flow=False
        )
        input_tensors = frozen_func.inputs
        output_tensors = frozen_func.outputs
        # Setting add_shapes=True causes shape inference and generates the `_output_shapes` attr
        graph_def = frozen_func.graph.as_graph_def(add_shapes=extra_shape_inference)

        # Now edit the attributes
        # - Optionally set `_output_quantized = True`
        # - Set batch dimension of `_output_shape` to 1
        for node in graph_def.node:
            # node.name is "model/tf_op_layer_..../..."
            # node.op is "LqceBconv2d64"
            # node.input is "model/.../add"
            # node.input is "model/.../filter"
            if node.op.startswith("LqceBconv2d"):

                if train_quantize or post_quantize:
                    node.attr["_output_quantized"].b = True
                else:
                    node.attr["_output_quantized"].b = False

                # Set batch dimension to 1
                # because wildcard dimensions are not supported for custom operators
                if extra_shape_inference:
                    node.attr["_output_shapes"].list.shape[0].dim[0].size = 1

                # Once we support fused activations, here we can
                # choose one of the activations in the enum `TfLiteFusedActivation`
                # defined in `lite/c/builtin_op_data.h`
                # "", "RELU", "RELU1", "RELU6"
                # node.attr["_fused_activation"].str = ""

                # Setting _output_types will be needed for bitpacked weights
                # node.attr['_output_types'].list.type.extend([
                #    types_pb2.DT_FLOAT,
                #    ])

        # Create the converter
        converter = tf.compat.v1.lite.TFLiteConverter(
            graph_def,
            input_tensors,
            output_tensors,
            experimental_debug_info_func=_build_debug_info_func(frozen_func.graph),
        )
        converter.allow_custom_ops = True

        if train_quantize:
            converter.inference_type = tf.uint8
            input_arrays = converter.get_input_arrays()
            # We usually have input images that are scaled to the range [-1,1],
            # so set the mean to 0 and standard deviation to 1
            converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
            converter.default_ranges_stats = (-3, 3)

        if post_quantize:
            if self.has_custom_ops:
                raise Exception(
                    "Post-training quantization is not supported for custom ops"
                )

            converter.representative_dataset = self.representative_dataset_gen
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # The lines below enforce full int8 (error on unsupported op)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        if new_converter:
            converter.experimental_enable_mlir_converter = True
            converter.experimental_new_converter = True
            if train_quantize or post_quantize:
                converter.experimental_new_quantizer = True

        return converter.convert()
