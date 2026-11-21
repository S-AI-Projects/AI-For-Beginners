# Execution Evidence for Issue #31 (R38-T031)

- **Student**: Tanishka .
- **Task**: R38-T031: cosine-similarity lookup — runnable concept example [lessons/5-NLP/14-Embeddings]
- **Repository**: https://github.com/S-AI-Projects/AI-For-Beginners
- **Branch**: `R38-T031`
- **Lab Directory**: `lessons/5-NLP/14-Embeddings/applied-labs/r38-t031`
- **Environment**: Python 3.15.2, git version 2.55.0

---

## 1. Automated Test Suite Execution

```bash
python3 -m unittest discover -s lessons/5-NLP/14-Embeddings/applied-labs/r38-t031 -p "test_*.py" -v
```

### Output:
```
test_arbitrary_vector_similarity (test_example.TestCosineSimilarityContract.test_arbitrary_vector_similarity)
Cosine similarity on arbitrary known non-trivial 2D vectors: [3, 4] vs [4, 3] = 24/25 = 0.96. ... ok
test_deliberate_wrong_result_fails (test_example.TestCosineSimilarityContract.test_deliberate_wrong_result_fails)
Verify that a deliberate incorrect expectation produces a test failure. ... ok
test_dimension_mismatch_raises_value_error (test_example.TestCosineSimilarityContract.test_dimension_mismatch_raises_value_error)
Vectors of differing lengths must raise ValueError. ... ok
test_dot_product_calculation (test_example.TestCosineSimilarityContract.test_dot_product_calculation)
dot_product returns correct sum of element-wise products. ... ok
test_empty_vectors_raise_value_error (test_example.TestCosineSimilarityContract.test_empty_vectors_raise_value_error)
Empty vectors must raise ValueError. ... ok
test_find_most_similar_ranking (test_example.TestCosineSimilarityContract.test_find_most_similar_ranking)
find_most_similar correctly ranks vocabulary candidates by similarity. ... ok
test_identical_vectors_yield_one (test_example.TestCosineSimilarityContract.test_identical_vectors_yield_one)
[1,0] versus [1,0] gives 1. ... ok
test_non_finite_values_raise_value_error (test_example.TestCosineSimilarityContract.test_non_finite_values_raise_value_error)
NaN and infinity values must raise ValueError. ... ok
test_non_numeric_types_raise_type_error (test_example.TestCosineSimilarityContract.test_non_numeric_types_raise_type_error)
Non-numeric or boolean values must raise TypeError. ... ok
test_opposite_vectors_yield_negative_one (test_example.TestCosineSimilarityContract.test_opposite_vectors_yield_negative_one)
Opposite direction vectors yield -1.0. ... ok
test_orthogonal_vectors_yield_zero (test_example.TestCosineSimilarityContract.test_orthogonal_vectors_yield_zero)
[1,0] versus [0,1] gives 0. ... ok
test_vector_norm_calculation (test_example.TestCosineSimilarityContract.test_vector_norm_calculation)
vector_norm returns correct Euclidean norm sqrt(sum(x_i^2)). ... ok
test_zero_vector_returns_undefined_sentinel (test_example.TestCosineSimilarityContract.test_zero_vector_returns_undefined_sentinel)
A zero vector returns the documented undefined-result sentinel (None). ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.001s

OK
```

---

## 2. Example Execution Output

```bash
python3 lessons/5-NLP/14-Embeddings/applied-labs/r38-t031/example.py
```

### Output:
```
============================================================
R38-T031: Cosine-Similarity Lookup Runnable Concept Example
============================================================
Identical vectors ([1, 0] vs [1, 0]):
  u = [1, 0], v = [1, 0] -> cosine_similarity = 1.0
Orthogonal vectors ([1, 0] vs [0, 1]):
  u = [1, 0], v = [0, 1] -> cosine_similarity = 0.0
Opposite vectors ([1, 0] vs [-1, 0]):
  u = [1, 0], v = [-1, 0] -> cosine_similarity = -1.0
Zero-norm vector vs [1, 0]:
  u = [1, 0], v = [0, 0] -> cosine_similarity = None
Both zero-norm vectors:
  u = [0, 0], v = [0, 0] -> cosine_similarity = None
Arbitrary vectors ([3, 4] vs [4, 3]):
  u = [3, 4], v = [4, 3] -> cosine_similarity = 0.96

Demonstrating Vocabulary Embedding Lookup:
Query: 'monarch' with vector [0.95, 0.05]
  1. king         : 0.9983
  2. queen        : 0.9841
  3. apple        : 0.0526
  4. zero_token   : UNDEFINED (sentinel None)
```

---

## 3. Deliberate Failure Verification

Running a test case with an incorrect expectation exits with code 1:
```
AssertionError: 1 != 2
FAILED (failures=1)
Exit code: 1
```
