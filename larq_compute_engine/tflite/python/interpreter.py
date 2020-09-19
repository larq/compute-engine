from typing import Iterator, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from larq_compute_engine.tflite.python import interpreter_wrapper_lite

__all__ = ["Interpreter"]

Data = Union[np.ndarray, List[np.ndarray]]


def data_generator(x: Data) -> Iterator[List[np.ndarray]]:
    if not isinstance(x, (list, np.ndarray)) or len(x) == 0:
        raise ValueError(
            "Expected either a non-empty list of inputs or a Numpy array with "
            f"implicit initial batch dimension. Received: {x}"
        )

    if isinstance(x, np.ndarray):
        x = [x]

    for inputs in zip(*x):
        yield [np.expand_dims(inp, axis=0) for inp in inputs]


class Interpreter:
    """Interpreter interface for Larq Compute Engine Models.

    !!! example
        ```python
        lce_model = convert_keras_model(model)
        interpreter = Interpreter(lce_model)
        interpreter.predict(input_data, verbose=1)
        ```

    # Arguments
        flatbuffer_model: A serialized Larq Compute Engine model in the flatbuffer format.

    # Attributes
        input_types: Returns a list of input types.
        input_shapes: Returns a list of input shapes.
        output_types: Returns a list of output types.
        output_shapes: Returns a list of output shapes.
    """

    def __init__(self, flatbuffer_model: bytes):
        self.interpreter = interpreter_wrapper_lite.LiteInterpreter(flatbuffer_model)

    @property
    def input_types(self) -> list:
        """Returns a list of input types."""
        return self.interpreter.input_types

    @property
    def input_shapes(self) -> List[Tuple[int]]:
        """Returns a list of input shapes."""
        return self.interpreter.input_shapes

    @property
    def output_types(self) -> list:
        """Returns a list of output types."""
        return self.interpreter.output_types

    @property
    def output_shapes(self) -> List[Tuple[int]]:
        """Returns a list of output shapes."""
        return self.interpreter.output_shapes

    def predict(self, x: Data, verbose: int = 0) -> Data:
        """Generates output predictions for the input samples.

        # Arguments
            x: Input samples. A Numpy array, or a list of arrays in case the model has
                multiple inputs.
            verbose: Verbosity mode, 0 or 1.

        # Returns
            Numpy array(s) of output predictions.
        """

        data_iterator = data_generator(x)
        if verbose >= 1:
            data_iterator = tqdm(data_iterator)

        prediction_iter = (self.interpreter.predict(inputs) for inputs in data_iterator)
        outputs = [np.concatenate(batches) for batches in zip(*prediction_iter)]

        if len(self.output_shapes) == 1:
            return outputs[0]
        return outputs
