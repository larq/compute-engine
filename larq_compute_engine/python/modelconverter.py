"""TF lite converter for larq models."""
import tensorflow as tf
import numpy as np
import larq as lq
from larq_compute_engine import bsign
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
        keras_result = "success"
        if tf_2_or_newer():
            session_result = "tensorflow 1.x only"
        else:
            session_result = "success"
            try:
                tflite_model = self.convert_sessionmethod()
            except Exception as e:
                session_result = str(e)

        try:
            tflite_model2 = self.convert_kerasmethod()
            if tflite_model is None:
                tflite_model = tflite_model2
        except Exception as e:
            keras_result = str(e)

        print("Conversion result:")
        print(f"Session method: {session_result}")
        print(f"Keras method: {keras_result}")

        if tflite_model is not None:
            print(f"Saving tf lite model as {filename}")
            if filename is not None:
                open(filename, "wb").write(tflite_model)
        else:
            print("Did not save tf lite model.")

        return tflite_model

    def fix_quantizers(self):
        result = True
        for l in self.model.layers:
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
            kernel_quantizer = None
            try:
                kernel_quantizer = l.kernel_quantizer
            except AttributeError:
                pass
            if kernel_quantizer is not None:
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
                    l.set_weights([means * np.sign(np.sign(w) + 0.5)])
                    # TODO: Implement the means in a multiplication layer after the convolution
                elif quantizer_replacements[name] is None:
                    print(f"ERROR: Kernel quantizer {name} not yet supported.")
                    result = False
                else:
                    l.kernel_quantizer = None
                    w = l.get_weights()[0]
                    # wbin = (w >= 0)
                    # wpacked = np.packbits(wbin, axis=3, bitorder='little')
                    binary_weights = np.sign(np.sign(w) + 0.5)
                    l.set_weights([binary_weights])
        return result

    def convert_kerasmethod(self):
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
