# Synthetic TEM Interface Example

This example uses generated arrays and simulated detector boxes to exercise the public windowing, source-coordinate merging, detector-box prompt schema, substrate-consensus, scale-verification, and projected-morphometry interfaces. It does not use a dissertation image or annotation and is not an accuracy benchmark.

From an installed repository:

```text
python examples/synthetic_tem/run_minimal_example.py --output path/to/example_output.json
```

The expected values are checked by `tests/test_synthetic_example.py`. Model-weight loading is tested separately because the release assets are deliberately kept outside Git.
