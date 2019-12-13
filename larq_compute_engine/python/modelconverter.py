"""TF lite converter for larq models."""
import tensorflow as tf
import numpy as np
import larq as lq
from larq_compute_engine import bsign, bconv2d64
from larq_compute_engine.python.utils import tf_2_or_newer
from tensorflow.keras.utils import get_custom_objects

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


def create_bconv_layer(weights, strides, padding, transpose=True, mul_weights=None):
    """
    Creates a binary convolution layer for tflite.

    If `transpose` is True, transposes from HWIO to OHWI

    When `mul_weights` is not `None`, it should be a 1D array of size equal to the filter out-channel dimension.
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

    if mul_weights is not None:
        if len(mul_weights.shape) != 1 or mul_weights.shape[0] != weights.shape[0]:
            print(
                f"ERROR: Argument mul_weights should have shape ({weights.shape[0]}) but has shape {mul_weights.shape}"
            )
            mul_weights = None

    def bconv_op(x):
        y = bconv2d64(
            x,
            weights,
            strides,
            padding,
            data_format="NHWC",
            filter_format=filter_format,
        )
        # Do the {0,1} -> {-1,1} transformation
        # If a,b are {-1,1} vectors then A = (a+1)/2 and B = (b+1)/2
        # are the corresponding {0,1} vectors.
        # We have
        #   dot(a,b) = n - 2 * XOR(A,B)
        # where n is the size of the vector.
        # The convolution will compute XOR(A,B) so we have to transform it back
        # One awesome property about this is that it already takes care of bitpadding,
        # so we can ignore that in the C++ code.

        # TODO: Detect if there is a batchnorm and merge this with that
        y = dotproduct_size - 2 * y

        if mul_weights is not None:
            y *= mul_weights
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
    It will remove the kernel quantizers and only store the signs instead of the latent weights.

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

    def convert(self, filename=None):
        """Convert and return the tflite model.

        Optionally save the model to a file.

        # Arguments
        filename: If `None`, then returns the tflite model object. If its a string then it saves the model to that filename.
        """
        if not self.fix_quantizers():
            print("Model contains unsupported quantizers. No conversion will be done.")
            return None

        tflite_model = None

        result_log = []
        result_summary = []

        if tf_2_or_newer():
            result_summary.append("Session method: Tensorflow 1.x only")
        else:
            try:
                tflite_model = self.convert_sessionmethod()
                result_summary.append("Session method: success")
            except Exception as e:
                result_log.append(f"Session method log:\n{str(e)}")
                result_summary.append("Session method: failed")

        try:
            tflite_model2 = self.convert_kerasmethod(new_converter=True)
            if tflite_model is None:
                tflite_model = tflite_model2
            result_summary.append("MLIR method: success")
        except Exception as e:
            result_log.append(f"MLIR method log:\n{str(e)}")
            result_summary.append("MLIR method: failed")

        try:
            tflite_model3 = self.convert_kerasmethod(new_converter=False)
            if tflite_model is None:
                tflite_model = tflite_model3
            result_summary.append("Keras method: success")
        except Exception as e:
            result_log.append(f"Keras method log:\n{str(e)}")
            result_summary.append("Keras method: failed")

        print("\n----------------\nConversion logs:")
        for log in result_log:
            print("----------------")
            print(log)

        print("----------------\nConversion summary:")
        for log in result_summary:
            print(log)

        if tflite_model is not None:
            if filename is not None:
                print(f"Saving tf lite model as {filename}")
                open(filename, "wb").write(tflite_model)
        else:
            print("Did not save tf lite model.")

        return tflite_model

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
                w = l.get_weights()[0]

                if isinstance(l, lq.layers.QuantConv2D):
                    if len(w.shape) != 4:
                        print(
                            f"ERROR: Weights of layer {l.name} have shape {w.shape} which does not have rank 4."
                        )
                        result = False
                    else:
                        # Create a new layer with those weights
                        bconvlayer = create_bconv_layer(
                            w,
                            l.strides,
                            l.padding,
                            transpose=True,
                            mul_weights=mul_weights,
                        )
                        replacement_dict[l.name] = bconvlayer
                else:
                    binary_weights = np.sign(np.sign(w) + 0.5)
                    l.set_weights([binary_weights])

        if result and replacement_dict:
            new_model = replace_layers(self.model, replacement_dict)
            if new_model is None:
                return False
            else:
                self.model = new_model

        return result

    def convert_kerasmethod(self, new_converter=False):
        """Conversion through the 'Keras method'

        This method works with many normal models. When adding a single Lambda layer, such as `tf.keras.layers.Lambda(tf.sign)` or with a custom op, then it still works.
        However, sometimes, when adding *more than one* of such layers, at any place in the network, then it stops working.
        """
        if tf_2_or_newer():
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        else:
            keras_file = "/tmp/modelconverter_temporary.h5"
            tf.keras.models.save_model(self.model, keras_file)
            converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
        converter.allow_custom_ops = True
        if new_converter:
            converter.experimental_enable_mlir_converter = True
            converter.experimental_new_converter = True
        return converter.convert()

    def convert_sessionmethod(self):
        """Conversion through the 'Session method'

        Unlike the Keras method, this one works with multiple Lambda layers with custom ops. However, it sometimes fails with BatchNormalization layers.

        Although it is a different error message, in the following issue it is suggested to replace `tf.keras.layers.BatchNormalization` by `tf.layers.batch_normalization(fused=False)`.
        https://github.com/tensorflow/tensorflow/issues/25301
        """
        converter = tf.lite.TFLiteConverter.from_session(
            tf.compat.v1.keras.backend.get_session(),
            self.model.inputs,
            self.model.outputs,
        )
        converter.allow_custom_ops = True
        return converter.convert()
