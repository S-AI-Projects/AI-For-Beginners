# Execution Evidence for Issue #22 (R38-T022)

- **Task**: Sangharsh satish sawant — R38-T022: token-budget preflight — boundary-case lesson extension [lessons/5-NLP/17-GenerativeNetworks]
- **Branch**: `Sangharshsatishsawant/e26b07f0602/R38-T022`
- **Lab Directory**: `lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t022`

## Test Results

Command:
```bash
python3 -m unittest discover -s lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t022 -p "test_*.py" -v
```

Output:
```
test_boundary_overflow_large (test_example.TestTokenBudgetPreflight.test_boundary_overflow_large)
Boundary overflow of many tokens over small budget. ... ok
test_boundary_overflow_single_token (test_example.TestTokenBudgetPreflight.test_boundary_overflow_single_token)
Boundary overflow of exactly 1 token over budget. ... ok
test_empty_input_retains_none (test_example.TestTokenBudgetPreflight.test_empty_input_retains_none)
Empty input string retains no tokens and reports 0 truncated tokens. ... ok
test_exact_budget_match (test_example.TestTokenBudgetPreflight.test_exact_budget_match)
Prompt length exactly equals budget; retains all, reports 0 truncated. ... ok
test_example_scenario_overflow (test_example.TestTokenBudgetPreflight.test_example_scenario_overflow)
Budget 3 with 'one two three four' retains ['one','two','three'] and reports 1 removed token. ... ok
test_negative_budget_rejected (test_example.TestTokenBudgetPreflight.test_negative_budget_rejected)
Negative budget is rejected with ValueError. ... ok
test_type_validation (test_example.TestTokenBudgetPreflight.test_type_validation)
Reject invalid types for prompt and budget. ... ok
test_under_budget (test_example.TestTokenBudgetPreflight.test_under_budget)
Prompt length is strictly less than budget; retains all, reports 0 truncated. ... ok
test_whitespace_only_input_retains_none (test_example.TestTokenBudgetPreflight.test_whitespace_only_input_retains_none)
Whitespace-only input string produces no tokens and reports 0 truncated tokens. ... ok
test_whitespace_tokenizer_teaching_approximation (test_example.TestTokenBudgetPreflight.test_whitespace_tokenizer_teaching_approximation)
Verify the deterministic whitespace tokenizer behavior and type handling. ... ok
test_zero_budget_boundary (test_example.TestTokenBudgetPreflight.test_zero_budget_boundary)
Zero budget retains 0 tokens and reports all original tokens as truncated. ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.001s

OK
```

## Example Run Output

Command:
```bash
python3 lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t022/example.py
```

Output:
```
=== Generative Networks: Token-Budget Preflight Demonstration ===
Prompt: 'one two three four', Budget: 3
  -> Retained tokens: ['one', 'two', 'three']
  -> Truncated count: 1
  -> Original count : 4

Prompt: 'alpha beta gamma', Budget: 3
  -> Retained tokens: ['alpha', 'beta', 'gamma']
  -> Truncated count: 0

Prompt: 'hello world', Budget: 0
  -> Retained tokens: []
  -> Truncated count: 2

Prompt: '', Budget: 5
  -> Retained tokens: []
  -> Truncated count: 0
```
