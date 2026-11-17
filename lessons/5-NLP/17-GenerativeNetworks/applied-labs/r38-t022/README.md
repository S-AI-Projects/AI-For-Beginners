# Applied Lab R38-T022: Token-Budget Preflight (Boundary-Case Lesson Extension)

## Overview & Lesson Concept Connection

In **Generative Networks** ([Lesson 17: Generative Networks](../../README.md)), text generation typically begins from a user-supplied **prompt**. Whether using Recurrent Neural Networks (RNNs/LSTMs/GRUs) or Transformer models, language models possess a finite input context window (maximum token budget).

If an incoming prompt exceeds the allocated token budget:
- Generating beyond memory and computational capacity causes sequence overflow, out-of-bounds indexing, or degraded generations.
- A **token-budget preflight** check inspects the tokenized prompt before generation begins, reporting retained tokens, truncated count, and enforcing boundary constraints.

---

## Teaching Approximation Notice

Real-world production models rely on subword tokenizers (such as Byte-Pair Encoding (BPE), WordPiece, or SentencePiece). This lab employs a **deterministic local whitespace tokenizer fixture**, clearly labeled as a teaching approximation, to isolate token-budget boundary mechanics cleanly without requiring external tokenization binaries or large vocabulary models.

---

## Boundary-Case Specifications

The preflight mechanism rigorously tests and handles the following boundary conditions:
1. **Standard Overflow:** Prompt tokens exceeding budget are truncated cleanly from the end, reporting retained tokens and truncated count (e.g. Budget 3 with `"one two three four"` retains `["one", "two", "three"]` and reports 1 truncated token).
2. **Exact Budget Match:** When prompt token count equals budget (e.g., 3 tokens with budget 3), all tokens are retained and truncated count is 0.
3. **Under Budget:** When token count is strictly less than budget, all tokens are retained with 0 truncated.
4. **Empty / Whitespace Input:** Prompts with no tokens retain an empty list `[]` and report 0 truncated tokens.
5. **Zero Budget (`budget = 0`):** An edge case where no prompt tokens may be ingested; retains `[]` and reports all prompt tokens as truncated.
6. **Negative Budget (`budget < 0`):** Strictly rejected with a `ValueError`.
7. **Type Validation:** Non-string prompts or non-integer budgets are rejected with a `TypeError`.

---

## Lab Files

- `example.py`: Implementation of `token_budget_preflight`, `whitespace_tokenizer`, and a runnable demonstration CLI.
- `test_example.py`: Complete unit test suite verifying all boundary conditions, error handling, and teaching fixtures.
- `README.md`: Architectural documentation and boundary case descriptions.

---

## Running the Code

### Execute Runnable Demonstration
```bash
python3 lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t022/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t022 -p "test_*.py"
```
