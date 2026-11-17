# Execution Evidence for Issue #21 (R38-T021)

- **Task**: Sangharsh satish sawant — R38-T021: tensor-shape contract — runnable concept example [lessons/4-ComputerVision/09-Autoencoders]
- **Branch**: `Sangharshsatishsawant/e26b07f0602/R38-T021`
- **Lab Directory**: `lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t021`

## Test Results

Command:
```bash
python3 -m unittest discover -s lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t021 -p "test_*.py" -v
```

Output:
```
test_multi_batch_preservation (test_example.TestTensorShapeContract.test_multi_batch_preservation)
Test with batch size N > 1 with rectangular dimensions (2, 3, 2). ... ok
test_rejection_greater_than_255_pixels (test_example.TestTensorShapeContract.test_rejection_greater_than_255_pixels)
Reject pixels > 255. ... ok
test_rejection_inconsistent_dimensions_ragged (test_example.TestTensorShapeContract.test_rejection_inconsistent_dimensions_ragged)
Reject ragged/inconsistent tensor dimensions. ... ok
test_rejection_invalid_rank (test_example.TestTensorShapeContract.test_rejection_invalid_rank)
Reject tensors that are not 3D (N x H x W). ... ok
test_rejection_negative_pixels (test_example.TestTensorShapeContract.test_rejection_negative_pixels)
Reject pixels < 0. ... ok
test_rejection_non_finite_float (test_example.TestTensorShapeContract.test_rejection_non_finite_float)
Reject float('nan') and float('inf'). ... ok
test_rejection_non_numeric_values (test_example.TestTensorShapeContract.test_rejection_non_numeric_values)
Reject non-numeric pixel values (strings, booleans, None, etc.). ... ok
test_rejection_non_sequence_input (test_example.TestTensorShapeContract.test_rejection_non_sequence_input)
Reject top-level non-sequence input. ... ok
test_rejection_zero_dimension (test_example.TestTensorShapeContract.test_rejection_zero_dimension)
Reject empty dimensions like (1, 0, 0). ... ok
test_shape_preservation_and_normalization (test_example.TestTensorShapeContract.test_shape_preservation_and_normalization)
Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0 -> 0.0 and 255 -> 1.0. ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.043s

OK
```

## Example Run Output

Command:
```bash
python3 lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t021/example.py
```

Output:
```
=== Autoencoder Input Preprocessing: Tensor-Shape Contract & Normalization ===
Original input batch: [[[0, 255], [128, 64]]]
Input shape: (1, 2, 2) (1 sample, height=2, width=2)
Normalized batch: [[[0.0, 1.0], [0.5019607843137255, 0.25098039215686274]]]
Normalized shape: (1, 2, 2) (Shape strictly preserved)
Contract verified successfully: pixel values in [0, 1] without altering shape.
```
