# Token-Budget Preflight — Failure and Recovery Example (R38-T048)

## 1. Task Objective

In Natural Language Processing (NLP) and Large Language Model (LLM) engineering, neural architectures have strict **maximum input sequence lengths** (context window limits). Sending text that exceeds these limits can cause server-side HTTP 400 errors, expensive token overuse, or silent model truncation that discards crucial instructions.

To prevent runtime failures, production client applications implement **preflight validation**: checking prompt lengths against a configured token budget locally before dispatching requests to remote model inference endpoints.

This applied software-engineering lab demonstrates:
1. A deterministic local token-budget preflight validator (`preflight_text`).
2. Truncation and token accounting (retained vs. removed token counts).
3. Fail-fast error rejection for invalid negative token budgets (`ValueError`).
4. A complete **failure and recovery sequence**: demonstrating how an application catches an invalid request and cleanly processes subsequent valid requests without state corruption.

---

## 2. Core Concepts

### What is Tokenization?
Neural networks cannot process raw characters directly; they operate on numeric tensors. Text must first be broken down into discrete chunks called **tokens**. While commercial LLMs use subword tokenization (such as Byte-Pair Encoding or WordPiece), this educational lab implements a deterministic **whitespace tokenizer** using Python's standard library.

### What is a Token Budget?
A token budget is a non-negative integer specifying the maximum allowable tokens for a given prompt, user message, or context window slice.

### How Truncation Works
When input token count exceeds the configured budget:
$$\text{original\_token\_count} = N$$
$$\text{retained\_tokens} = \text{tokens}[0 : \text{budget}]$$
$$\text{retained\_count} = \text{budget}$$
$$\text{removed\_count} = N - \text{budget}$$
$$\text{is\_truncated} = \text{True}$$

Across all inputs and configurations, the mathematical sum invariant holds:
$$\text{retained\_count} + \text{removed\_count} = \text{original\_token\_count}$$

---

## 3. Mandatory Specification Fixtures

### Case 1: Normal Truncation
- **Input Text:** `"one two three four"`
- **Configured Budget:** `3`
- **Original Tokens:** `["one", "two", "three", "four"]` ($N = 4$)
- **Retained Tokens:** `["one", "two", "three"]` ($\text{count} = 3$)
- **Removed Count:** $1$
- **Is Truncated:** `True`

### Case 2: Empty Input
- **Input Text:** `""` (or whitespace-only `"   "`)
- **Configured Budget:** `3`
- **Original Tokens:** `[]` ($N = 0$)
- **Retained Tokens:** `[]` ($\text{count} = 0$)
- **Removed Count:** $0$
- **Is Truncated:** `False`
- *Note:* Empty input is a valid non-exceptional case and returns 0 tokens without error.

### Case 3: Negative Budget (Fail-Fast Rejection)
- **Input Text:** `"one two"`
- **Configured Budget:** `-1`
- **Result:** Rejects immediately by raising `ValueError("Token budget cannot be negative, got -1")`.
- *Note:* Negative budgets are physically meaningless. The validator does not silently clamp negative values to 0.

### Case 4: Failure and Recovery Demonstration
The central contribution demonstrates clean client-side recovery:
1. **Invalid Attempt:** The client submits `text="one two"`, `budget=-1`.
2. **Error Caught:** The validator raises `ValueError`, which is intercepted and logged.
3. **Recovery Request:** The client submits a valid request: `text="one two three four"`, `budget=3`.
4. **Successful Output:** The validator processes the valid request cleanly, producing retained tokens `["one", "two", "three"]` with removed count $1$.
5. **State Isolation:** The preceding failure does not leak state or disrupt subsequent execution.

---

## 4. Boundary Cases

1. **Budget Equal to Token Count:** `"apple banana"`, budget `2` $\implies$ retains all 2 tokens, 0 removed, `is_truncated=False`.
2. **Budget Exceeds Token Count:** `"hello"`, budget `10` $\implies$ retains all tokens, 0 removed, `is_truncated=False`.
3. **Budget Zero ($0$):** `"alpha beta"`, budget `0` $\implies$ retains `[]`, 2 removed, `is_truncated=True`.
4. **Multiple and Irregular Whitespace:** `"one   two\tthree\nfour"` $\implies$ correctly parsed into 4 tokens without empty strings.
5. **Type Safety:** Rejects non-string texts and non-integer budgets. Rejects Python `bool` values (`True`/`False`) with `TypeError` because booleans are not valid token counts.

---

## 5. Scope and Educational Limitations

This is a local mathematical and software demonstration. It does **not**:
1. Load or train a neural language model or subword tokenizer (e.g. Tiktoken, SentencePiece, HuggingFace Tokenizers).
2. Claim that whitespace token counts match subword token counts in commercial LLMs (e.g., punctuation, compound words, and code often split into multiple subwords in real tokenizers).
3. Connect to external cloud APIs (e.g., OpenAI, Azure, Anthropic).
4. Require GPUs, paid services, or hardware accelerators.

---

## 6. Execution Instructions

### Prerequisites
- Python 3.8+ (standard library only; no external dependencies required).

### Run the Example
```bash
python3 lessons/5-NLP/13-TextRep/applied-labs/r38-t048/example.py
```

### Run Automated Tests
```bash
python3 -m unittest discover -v -s lessons/5-NLP/13-TextRep/applied-labs/r38-t048 -p "test_*.py"
```

### Expected Output
```
=================================================================
Token-Budget Preflight — Failure and Recovery Example (R38-T048)
=================================================================

--- Case 1: Normal Truncation ---
Input text:           'one two three four'
Configured budget:    3
Original tokens:      ['one', 'two', 'three', 'four']
Original token count: 4
Retained tokens:      ['one', 'two', 'three']
Retained count:       3
Removed count:        1
Is truncated:         True
Invariant check:      3 + 1 == 4 (PASS)

--- Case 2: Empty Input ---
Input text:           ''
Configured budget:    3
Original tokens:      []
Original token count: 0
Retained tokens:      []
Retained count:       0
Removed count:        0
Is truncated:         False
Invariant check:      0 + 0 == 0 (PASS)

--- Case 3 & 4: Failure and Recovery Demonstration ---
Attempting invalid request: text='one two', budget=-1
Caught expected exception: ValueError: Token budget cannot be negative, got -1
Explanation: Negative budgets are physically meaningless and rejected fail-fast.

Recovering with valid request: text='one two three four', budget=3...
Recovery succeeded! Retained tokens: ['one', 'two', 'three']
Retained count: 3, Removed count: 1

--- Boundary Case: Zero Budget (Budget = 0) ---
Input text:           'alpha beta gamma'
Configured budget:    0
Original tokens:      ['alpha', 'beta', 'gamma']
Original token count: 3
Retained tokens:      []
Retained count:       0
Removed count:        3
Is truncated:         True
Invariant check:      0 + 3 == 3 (PASS)
```

