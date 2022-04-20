from typing import Iterator, List, Tuple, Union, Optional

import numpy as np
from tqdm import tqdm

Data = Union[np.ndarray, List[np.ndarray]]


def data_generator(x: Union[Data, Iterator[Data]]) -> Iterator[List[np.ndarray]]:
    if isinstance(x, np.ndarray):
        for inputs in x:
            yield [np.expand_dims(inputs, axis=0)]
    elif isinstance(x, list):
        for inputs in zip(*x):
            yield [np.expand_dims(inp, axis=0) for inp in inputs]
    elif hasattr(x, "__next__") and hasattr(x, "__iter__"):
        for inputs in x:
            if isinstance(inputs, np.ndarray):
                yield [np.expand_dims(inputs, axis=0)]
            else:
                yield [np.expand_dims(inp, axis=0) for inp in inputs]
    else:
        raise ValueError(
            "Expected either a list of inputs or a Numpy array with implicit initial "
            f"batch dimension or an iterator yielding one of the above. Received: {x}"
        )


class InterpreterBase:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    @property
    def input_types(self) -> list:
        """Returns a list of input types."""
        return self.interpreter.input_types

    @property
    def input_shapes(self) -> List[Tuple[int]]:
        """Returns a list of input shapes."""
        return self.interpreter.input_shapes

    @property
    def input_scales(self) -> List[Optional[Union[float, List[float]]]]:
        """Returns a list of input scales."""
        return self.interpreter.input_scales

    @property
    def input_zero_points(self) -> List[Optional[int]]:
        """Returns a list of input zero points."""
        return self.interpreter.input_zero_points

    @property
    def output_types(self) -> list:
        """Returns a list of output types."""
        return self.interpreter.output_types

    @property
    def output_shapes(self) -> List[Tuple[int]]:
        """Returns a list of output shapes."""
        return self.interpreter.output_shapes

    @property
    def output_scales(self) -> List[Optional[Union[float, List[float]]]]:
        """Returns a list of input scales."""
        return self.interpreter.output_scales

    @property
    def output_zero_points(self) -> List[Optional[int]]:
        """Returns a list of input zero points."""
        return self.interpreter.output_zero_points

    def predict(self, x: Union[Data, Iterator[Data]], verbose: int = 0) -> Data:
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
