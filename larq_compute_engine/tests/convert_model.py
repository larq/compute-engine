"""Convert model script.

This can be used to test model conversion via the CLI. Callers should overwrite
`model_fn` to return a Keras model to be converted.

Usage Examples:
- bazelisk test larq_compute_engine/tests:convert_model
- bazelisk test larq_compute_engine/tests:convert_model --test_arg="--outfile=/tmp/model.tflite"
- bazelisk run larq_compute_engine/tests:convert_model -- --outfile=/tmp/model.tflite
"""

import click

from larq_compute_engine.mlir.python.converter import convert_keras_model


def model_fn():
    raise NotImplementedError(
        "No model defined. This function should be overwritten by caller."
    )


@click.command()
@click.option(
    "--outfile",
    default="model.tflite",
    help="Destination used to save converted TFLite flatbuffer.",
    type=click.Path(writable=True, resolve_path=True),
)
def convert_model(outfile):
    model_lce = convert_keras_model(
        model_fn(), experimental_enable_bitpacked_activations=True
    )
    with open(outfile, "wb") as f:
        f.write(model_lce)

    click.secho(f"TFLite flatbuffer saved to '{outfile}'.")


if __name__ == "__main__":
    convert_model()
