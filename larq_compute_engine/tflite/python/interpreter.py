from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from larq_compute_engine.tflite.python import interpreter_wrapper_lite

__all__ = ["Interpreter"]


def normalize_input_data(input_data):
    """Normalise input data into a list (batch dimension) of lists (number of inputs) of
    numpy arrays."""
    # For a single-input model, we accept a single numpy array where the first
    # dimension is implicitly the batch dimension.
    if isinstance(input_data, np.ndarray):
        return [[np.expand_dims(data, axis=0)] for data in input_data]
    if not isinstance(input_data, list) or len(input_data) == 0:
        raise TypeError(
            "Expected either a non-empty list of inputs or a numpy array with "
            f"implicit initial batch dimension. Received: {input_data}"
        )
    # If the input data is not a list of lists assume that the model has a
    # single input and wrap each element in a singleton list.
    if not isinstance(input_data[0], list):
        return [[data] for data in input_data]
    return input_data


class Interpreter:
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

    def predict(
        self,
        input_data: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
        verbose: int = 0,
    ) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """Generates output predictions for the input samples.

        # Arguments
            input_data: Input samples.
            verbose: Verbosity mode, 0 or 1.

        # Returns
            A list of output predictions.
        """
        input_data = normalize_input_data(input_data)

        if verbose >= 1:
            input_data = tqdm(input_data)

        if len(self.output_shapes) == 1:
            return [self.interpreter.predict(data)[0] for data in input_data]

        return [self.interpreter.predict(data) for data in input_data]
