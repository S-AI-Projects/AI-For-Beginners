# Execution Evidence for Issue #27 (R38-T027)

- **Task**: Sangharsh satish sawant — R38-T027: cosine-similarity lookup — worked example with checks [lessons/5-NLP/14-Embeddings]
- **Branch**: `Sangharshsatishsawant/e26b07f0602/R38-T027`
- **Lab Directory**: `lessons/5-NLP/14-Embeddings/applied-labs/r38-t027`

## Test Results

Command:
```bash
python3 -m unittest discover -s lessons/5-NLP/14-Embeddings/applied-labs/r38-t027 -p "test_*.py" -v
```

Output:
```
test_dimension_mismatch_rejection (test_example.TestCosineSimilarityLookup.test_dimension_mismatch_rejection)
Mismatched vector lengths raise ValueError. ... ok
test_empty_vector_rejection (test_example.TestCosineSimilarityLookup.test_empty_vector_rejection)
Empty vectors raise ValueError. ... ok
test_find_most_similar_batch (test_example.TestCosineSimilarityLookup.test_find_most_similar_batch)
Batch lookup across candidate embeddings. ... ok
test_identical_vectors (test_example.TestCosineSimilarityLookup.test_identical_vectors)
[1,0] vs [1,0] = 1.0. ... ok
test_non_finite_rejection (test_example.TestCosineSimilarityLookup.test_non_finite_rejection)
Reject float('nan') or float('inf'). ... ok
test_opposite_vectors (test_example.TestCosineSimilarityLookup.test_opposite_vectors)
Opposite direction vectors yield -1.0. ... ok
test_orthogonal_vectors (test_example.TestCosineSimilarityLookup.test_orthogonal_vectors)
[1,0] vs [0,1] = 0.0. ... ok
test_type_validation (test_example.TestCosineSimilarityLookup.test_type_validation)
Non-numeric elements or non-sequence types raise TypeError. ... ok
test_worked_example_arithmetic_regression_check (test_example.TestCosineSimilarityLookup.test_worked_example_arithmetic_regression_check)
Regression check for documented worked example: ... ok
test_zero_vector_sentinel (test_example.TestCosineSimilarityLookup.test_zero_vector_sentinel)
Explicitly handle zero vectors by returning documented sentinel None. ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.001s

OK
```

## Example Run Output

Command:
```bash
python3 lessons/5-NLP/14-Embeddings/applied-labs/r38-t027/example.py
```

Output:
```
=== Word Embeddings: Cosine Similarity Lookup ===
Worked Example: u=[3, 4], v=[4, 3]
  -> Cosine Similarity: 0.96 (Expected: 0.96)
[1, 0] vs [1, 0]: 1.0
[1, 0] vs [0, 1]: 0.0
[1, 0] vs [0, 0]: None (Zero-vector sentinel None)
```
