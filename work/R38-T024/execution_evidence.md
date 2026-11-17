# Execution Evidence for Issue #24 (R38-T024)

- **Task**: Sangharsh satish sawant — R38-T024: prompt-template escaping — invariant and state trace [lessons/5-NLP/20-LangModels]
- **Branch**: `Sangharshsatishsawant/e26b07f0602/R38-T024`
- **Lab Directory**: `lessons/5-NLP/20-LangModels/applied-labs/r38-t024`

## Test Results

Command:
```bash
python3 -m unittest discover -s lessons/5-NLP/20-LangModels/applied-labs/r38-t024 -p "test_*.py" -v
```

Output:
```
test_empty_template_and_variables (test_example.TestPromptTemplateEngine.test_empty_template_and_variables)
Empty template with empty variables dict succeeds cleanly. ... ok
test_reject_missing_keys (test_example.TestPromptTemplateEngine.test_reject_missing_keys)
Reject template execution when required keys are missing. ... ok
test_reject_unknown_keys (test_example.TestPromptTemplateEngine.test_reject_unknown_keys)
Reject template execution when unknown extraneous keys are supplied. ... ok
test_render_with_declared_variables (test_example.TestPromptTemplateEngine.test_render_with_declared_variables)
Render templates from declared variables. ... ok
test_state_trace_and_invariants (test_example.TestPromptTemplateEngine.test_state_trace_and_invariants)
Test that state trace is recorded and satisfies step invariants. ... ok
test_template_escaped_braces (test_example.TestPromptTemplateEngine.test_template_escaped_braces)
Template literal escaped braces {{ and }} are unescaped to { and }. ... ok
test_type_rejection (test_example.TestPromptTemplateEngine.test_type_rejection)
Reject invalid template or variable types. ... ok
test_user_data_braces_remain_literal_no_secondary_injection (test_example.TestPromptTemplateEngine.test_user_data_braces_remain_literal_no_secondary_injection)
Braces inside user data remain literal and are not recursively substituted. ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.001s

OK
```

## Example Run Output

Command:
```bash
python3 lessons/5-NLP/20-LangModels/applied-labs/r38-t024/example.py
```

Output:
```
=== Prompt-Template Escaping Engine & State Trace ===

--- Rendered Prompt ---
System: You are a helpful coding assistant.
User Query: Translate this snippet: if (x > 0) { return {key: 'val'}; }
Format: JSON

--- State Trace & Invariants ---
Step 1: parse_placeholders
  Invariant: All extracted placeholders are alphanumeric variable identifiers.
  State: {'template': 'System: {system_instruction}\nUser Query: {user_query}\nFormat: {output_format}', 'placeholders': ['system_instruction', 'user_query', 'output_format'], 'unique_count': 3}
Step 2: validate_keys
  Invariant: Provided keys match template declared placeholders exactly (no missing, no unknown).
  State: {'provided_keys': ['output_format', 'system_instruction', 'user_query'], 'matched_keys': ['output_format', 'system_instruction', 'user_query']}
Step 3: bind_literals
  Invariant: User variables bound as literal strings without evaluation of inner expressions.
  State: {'string_values': {'system_instruction': 'You are a helpful coding assistant.', 'user_query': "Translate this snippet: if (x > 0) { return {key: 'val'}; }", 'output_format': 'JSON'}, 'value_lengths': {'system_instruction': 35, 'user_query': 59, 'output_format': 4}}
Step 4: substitute
  Invariant: Single-pass substitution complete; user braces remain literal without secondary injection.
  State: {'rendered_prompt': "System: You are a helpful coding assistant.\nUser Query: Translate this snippet: if (x > 0) { return {key: 'val'}; }\nFormat: JSON", 'output_length': 128}

[SECURITY NOTICE]:
Syntactic escaping guarantees braces in user data are rendered literally (no secondary injection).
Note: Templating alone does NOT prevent semantic prompt injection (e.g. adversarial instructions).
```
