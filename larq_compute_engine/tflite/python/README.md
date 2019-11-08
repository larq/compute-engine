# Unit tests for TF Lite

The unit tests require the Larq Compute Engine python package (non-lite version). Please look at the [readme](../../../README.md) on how to build and install the python package.

They also require the Larq Compute Engine Lite python package. See the [TF lite build readme](../build/README.md) for instructions on how to build the TF lite package.

Once both packages have been installed, the unit tests can be evaluated using

```bash
cd larq_compute_engine/tflite/python
python3 -m pytest . -n auto
```
