"""TF lite model conversion."""
import tensorflow as tf


class ModelConverter:
    """Converter to create TF lite models from Keras models

    This is a simple wrapper around the existing TF lite converter which will try multiple methods.

    When the conversion fails, it can help to use the pip package `tf-nightly`.

    # Arguments
    model: The Keras model to convert.

    !!! example
        ```python
        import larq_compute_engine as lqce
        from larq_zoo import BiRealNet
        model = BiRealNet()
        conv = ModelConverter(model)
        conv.convert("birealnet.tflite")
        ```
    """

    def __init__(self, model):
        self.model = model

    def convert(self, filename):
        tflite_model = None
        keras_result = "success"
        session_result = "success"
        try:
            tflite_model = self.convert_kerasmethod()
        except Exception as e:
            keras_result = str(e)

        try:
            tflite_model2 = self.convert_sessionmethod()
            if tflite_model is None:
                tflite_model = tflite_model2
        except Exception as e:
            session_result = str(e)

        print("Converion result:")
        print("Keras method: {}".format(keras_result))
        print("Session method: {}".format(session_result))

        if tflite_model is not None:
            print("Saving tf lite model as {}".format(filename))
            open(filename, "wb").write(tflite_model)
            return True
        else:
            print("Did not save tf lite model.")
            return False

    def convert_kerasmethod(self):
        """Conversion through the 'Keras method'

        This method works with many normal models. When adding a single Lambda layer, such as `tf.keras.layers.Lambda(tf.sign)` or with a custom op, then it still works.
        However, sometimes, when adding *more than one* of such layers, at any place in the network, then it stops working.
        """
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
            tf.keras.backend.get_session(), self.model.inputs, self.model.outputs
        )
        converter.allow_custom_ops = True
        return converter.convert()
