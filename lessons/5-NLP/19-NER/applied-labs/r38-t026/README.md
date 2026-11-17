# Applied Lab R38-T026: Token-Budget Preflight — Refactoring the Teaching Example

## Overview & Lesson Concept Connection

In **Named Entity Recognition** ([Lesson 19: Named Entity Recognition](../../README.md)), text inputs are evaluated via token classification (such as BIO sequence tagging). Token classification models built on recurrent architectures (LSTMs) or transformer architectures operate with fixed-size sequence inputs.

When an incoming sentence or document exceeds the model's supported token capacity:
- A token budget preflight check ensures excess tokens are truncated while recording the retained tokens and truncated counts.
- Negative budgets are rejected, and empty inputs are handled cleanly without error.

---

## Refactoring the Teaching Example

This lab demonstrates modern software design by contrasting two architectures:

1. **Baseline Implementation (`token_budget_preflight_baseline`)**:
   - Monolithic procedural function handling input validation, whitespace tokenization, truncation logic, and report construction in one block.
2. **Refactored Architecture (`token_budget_preflight` + `TokenTruncationPolicy`)**:
   - Extracts a single named responsibility: `TokenTruncationPolicy`.
   - Isolates budget validation, boundary checks, and slice calculations into a reusable policy component.
   - Decouples policy enforcement from token extraction and report structuring.

### Strict Functional Equivalence
The test suite explicitly verifies **100% before-and-after behavioral equivalence** between the baseline implementation and the refactored policy across multiple test scenarios (empty strings, zero budgets, exact matches, under-budget inputs, and large inputs).

---

## Lab Files

- `example.py`: Contains both the baseline implementation and the refactored `TokenTruncationPolicy` pattern, with a runnable comparison demonstration CLI.
- `test_example.py`: Comprehensive test suite verifying truncation accuracy, error handling, boundary conditions, and strict functional equivalence.
- `README.md`: Lab architectural documentation.

---

## Running the Code

### Execute Runnable Demonstration
```bash
python3 lessons/5-NLP/19-NER/applied-labs/r38-t026/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/5-NLP/19-NER/applied-labs/r38-t026 -p "test_*.py"
```
