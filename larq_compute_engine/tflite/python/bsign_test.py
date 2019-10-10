import tensorflow as tf
import larq_compute_engine as lqce
from inference_test_base import InferenceTest


class BsignTest(InferenceTest):
    def test_bsign(self):
        img_input = tf.keras.layers.Input(shape=(28, 28, 3))
        out = lqce.bsign(img_input)
        model = tf.keras.Model(inputs=img_input, outputs=out)
        self.run_inference(model)


if __name__ == "__main__":
    tf.test.main()
