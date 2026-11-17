# Applied Lab R38-T025: Prompt-Template Escaping — Offline Service Adapter

## Overview & Lesson Concept Connection

In **Large Language Models** ([Lesson 20: Pre-Trained Large Language Models](../../README.md)), real-world generative applications communicate with hosted language model services (such as Azure OpenAI Service or OpenAI API) by structuring user tasks with prompt templates and dispatching requests over HTTP APIs.

In local development, automated testing, or sandboxed environments:
- Remote network dependencies are fragile, introduce latency, and may incur costs or access restrictions.
- An **offline service adapter** provides an architectural boundary: it models the client-service contract locally, validating request payload construction (model name, prompt string, temperature, token limits) and generating structured model responses without making external network calls.

---

## Architectural Components

### 1. Safe Prompt Template Rendering (`render_template`)
- Ingests template strings containing named placeholders `{variable_name}` and declared inputs.
- **Strict Key Validation:** Detects and rejects missing variables (`MissingKeyError`).
- **Literal Value Isolation:** Employs single-pass substitution guaranteeing that any braces `{...}` contained inside user variable strings are never recursively parsed as secondary template tags.
- **Literal Brace Escaping:** Unescapes double braces `{{` and `}}` to literal `{` and `}` without conflicting with variable substitution.

### 2. Offline Service Adapter (`OfflineModelAdapter`)
- Simulates the model dispatch interface locally without remote network dependencies.
- Enforces payload validation on `ModelRequest`:
  - Non-empty prompt strings.
  - Temperature range verification ($0.0 \le \text{temperature} \le 2.0$).
  - Strictly positive token allocations (`max_tokens > 0`).
  - Model identifier validation.
- Returns structured `ModelResponse` containing output text, response ID, stop finish reason, usage token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`), and an audit copy of the dispatched request.

---

## Lab Files

- `example.py`: Implementation of `render_template`, `MissingKeyError`, `ModelRequest`, `ModelResponse`, `OfflineModelAdapter`, and a runnable demonstration CLI.
- `test_example.py`: Comprehensive test suite verifying safe template rendering, key validation, literal brace preservation, payload validation, and end-to-end adapter dispatch.
- `README.md`: Lab architectural documentation.

---

## Running the Code

### Execute Runnable Demonstration
```bash
python3 lessons/5-NLP/20-LangModels/applied-labs/r38-t025/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/5-NLP/20-LangModels/applied-labs/r38-t025 -p "test_*.py"
```
