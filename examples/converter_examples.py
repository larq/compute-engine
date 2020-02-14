"""Examples of TF lite model conversion."""
import tensorflow as tf
import larq_compute_engine as lce
import larq_zoo as lqz

# Example of converting a model from Larq Zoo to TF lite
model = lqz.BinaryResNetE18(weights="imagenet")
converted = lce.convert_keras_model(model)
with open("binaryresnete18.tflite", "wb") as f:
    f.write(converted)

# Example of converting an h5 file
model = tf.keras.models.load_model("my_model.h5")
converted = lce.convert_keras_model(model)
with open("my_model.tflite", "wb") as f:
    f.write(converted)
