"""TF lite converter wrapper."""
import tensorflow as tf
import numpy as np
from larq_compute_engine import bsign
from tensorflow.keras.utils import get_custom_objects
from distutils.version import LooseVersion

get_custom_objects()["bsign"] = bsign


def tf_2_or_newer():
    return LooseVersion(tf.__version__) >= LooseVersion("2.0")


class ModelConverter:
    """Converter to create TF lite models from Keras models

    This is a simple wrapper around the existing TF lite converter which will try multiple methods.

    When the conversion fails, it can help to use the pip package `tf-nightly`.

    # Arguments
    model: The Keras model to convert.

    !!! example
        ```python
        from larq_zoo import BiRealNet
        model = BiRealNet()
        conv = ModelConverter(model)
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
        self.fix_quantizers()
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
        print("Session method: {}".format(session_result))
        print("Keras method: {}".format(keras_result))

        if tflite_model is not None:
            print("Saving tf lite model as {}".format(filename))
            if filename is not None:
                open(filename, "wb").write(tflite_model)
        else:
            print("Did not save tf lite model.")

        return tflite_model

    def fix_quantizers(self):
        for l in self.model.layers:
            try:
                if l.input_quantizer is not None:
                    l.input_quantizer = bsign
            except AttributeError:
                pass
            try:
                if l.kernel_quantizer is not None:
                    l.kernel_quantizer = None
                    l.set_weights(np.sign(np.sign(l.get_weights()) + 0.5))
            except AttributeError:
                pass

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
