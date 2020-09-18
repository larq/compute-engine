from typing import List, Union

import numpy as np
from tqdm import tqdm

from larq_compute_engine.tflite.python import interpreter_wrapper_lite

__all__ = ["Interpreter"]


class Interpreter(interpreter_wrapper_lite.LiteInterpreter):
    """Interpreter interface for Larq Compute Engine Models.

    !!! example
        ```python
        tflite_model = convert_keras_model(model)
        interpreter = Interpreter(tflite_model)
        interpreter.predict(input_data, verbose=1)
        ```

    # Arguments
        flatbuffer_model: A serialized TFLite model in the flatbuffer format.
    """

    def predict(
        self, x: Union[np.ndarray, List[np.ndarray]], verbose: int = 0
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generates output predictions for the input samples.

        # Arguments
            x: Input samples. A Numpy array, or a list of arrays in case the model has
                multiple inputs.
            verbose: Verbosity mode, 0 or 1.

        # Returns
            Numpy array(s) of output predictions.
        """

        if not isinstance(x, (list, np.ndarray)) or len(x) == 0:
            raise ValueError(
                "Expected either a non-empty list of inputs or a numpy array with "
                f"implicit initial batch dimension. Received: {x}"
            )

        if len(self.input_shapes) == 1:
            x = [x]

        batch_iter = tqdm(zip(*x)) if verbose >= 1 else zip(*x)
        prediction_batch_iter = (
            self.predict_batch([np.expand_dims(inp, axis=0) for inp in inputs])
            for inputs in batch_iter
        )
        outputs = [np.concatenate(batches) for batches in zip(*prediction_batch_iter)]

        if len(self.output_shapes) == 1:
            return outputs[0]
        return outputs
