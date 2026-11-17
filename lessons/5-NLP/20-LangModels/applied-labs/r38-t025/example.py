"""
Prompt-Template Escaping with Offline Service Adapter.

Lesson Concept Connection:
In Large Language Models (lessons/5-NLP/20-LangModels), applications interact with
models via service APIs (such as Azure OpenAI Service or OpenAI API) using prompt
templates. In testing, offline development, or CI environments, network calls to remote
APIs are undesirable or prohibited.

This module implements:
1. Safe prompt template rendering:
   - Renders variables into template placeholders `{key}` without remote dependencies.
   - Rejects missing keys (`MissingKeyError`).
   - Ensures braces `{...}` inside variables remain literal (preventing secondary template injection).
2. Offline Service Adapter (`OfflineModelAdapter`):
   - Mocks the model dispatch interface locally without network dependencies.
   - Validates request payload construction (model name, prompt string, temperature, max_tokens).
   - Generates deterministic simulated responses with structured metadata (usage tokens, finish reason).
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional


class TemplateError(Exception):
    """Base exception for prompt template errors."""
    pass


class MissingKeyError(TemplateError):
    """Raised when a template placeholder is missing from the variables dictionary."""
    pass


def extract_placeholders(template: str) -> List[str]:
    """Extract unescaped {identifier} placeholders in order."""
    pattern = re.compile(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})')
    return pattern.findall(template)


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Safely renders variables into template placeholders.

    - Rejects missing keys with MissingKeyError.
    - Preserves literal braces inside variable values via single-pass substitution.
    - Unescapes double braces {{ and }} to { and }.
    """
    if not isinstance(template, str):
        raise TypeError(f"Template must be a string, got {type(template).__name__}")
    if not isinstance(variables, dict):
        raise TypeError(f"Variables must be a dict, got {type(variables).__name__}")

    placeholders = set(extract_placeholders(template))
    missing = placeholders - set(variables.keys())
    if missing:
        raise MissingKeyError(f"Missing required template variable(s): {sorted(missing)}")

    # Convert values to strings safely
    str_vars: Dict[str, str] = {}
    for k, v in variables.items():
        if isinstance(v, (str, int, float, bool)):
            str_vars[k] = str(v)
        else:
            raise TypeError(f"Variable '{k}' has unsupported type {type(v).__name__}")

    # Placeholder replacement using unique UUID sentinel tokens to avoid collisions with {{ and }}
    import uuid
    escaped_left = f"__ESC_L_{uuid.uuid4().hex}__"
    escaped_right = f"__ESC_R_{uuid.uuid4().hex}__"

    temp = template.replace("{{", escaped_left).replace("}}", escaped_right)
    pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

    def replacer(match: re.Match) -> str:
        return str_vars[match.group(1)]

    # Single-pass substitution prevents secondary injection
    substituted = pattern.sub(replacer, temp)
    return substituted.replace(escaped_left, "{").replace(escaped_right, "}")


@dataclass(frozen=True)
class ModelRequest:
    """
    Standardized payload for dispatching a prompt to a language model service.
    """
    prompt: str
    model: str = "offline-gpt-3.5-mock"
    temperature: float = 0.7
    max_tokens: int = 150
    stop: Optional[List[str]] = None


@dataclass(frozen=True)
class TokenUsage:
    """Token usage metadata."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    """
    Standardized return structure from the language model service.
    """
    id: str
    model: str
    output_text: str
    finish_reason: str
    usage: TokenUsage
    request_payload: ModelRequest


class OfflineModelAdapter:
    """
    Offline service adapter that simulates the model dispatch interface locally.

    Features:
    - Zero remote network dependencies.
    - Strict payload verification (types, bounds for temperature, prompt validity).
    - Deterministic completion generation derived from the prompt hash or canned responses.
    - Full telemetry/audit log tracking all dispatched calls for assertion in tests.
    """

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response
        self.call_history: List[ModelRequest] = []

    def validate_request(self, request: ModelRequest) -> None:
        """Validates payload construction before simulation."""
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not (0.0 <= request.temperature <= 2.0):
            raise ValueError(f"Temperature {request.temperature} out of bounds [0.0, 2.0].")
        if request.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {request.max_tokens}.")
        if not isinstance(request.model, str) or not request.model:
            raise ValueError("Model must be a valid non-empty string identifier.")

    def dispatch(self, request: ModelRequest) -> ModelResponse:
        """
        Dispatches request through the offline adapter.
        Constructs and returns a structured ModelResponse.
        """
        self.validate_request(request)
        self.call_history.append(request)

        # Deterministic simulation
        if self.canned_response is not None:
            completion = self.canned_response
        else:
            # Deterministic pseudo-completion based on prompt
            prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:8]
            completion = f"[OfflineModelResponse for model={request.model} hash={prompt_hash}]"

        # Approximate token counts
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(completion.split())

        response_id = f"mock-resp-{len(self.call_history):04d}"

        return ModelResponse(
            id=response_id,
            model=request.model,
            output_text=completion,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            request_payload=request,
        )


def main() -> None:
    """Demonstrate safe prompt rendering and offline service adapter dispatch."""
    print("=== Prompt-Template Escaping & Offline Service Adapter ===")

    template = (
        "You are an assistant. Task: {task}\n"
        "Input Data: {raw_data}\n"
        "Guideline: Output format must match: {{'status': 'complete'}}"
    )
    variables = {
        "task": "Extract entity details",
        # Variables with literal braces must remain untouched
        "raw_data": "{'entity': 'Item_42', 'config': { 'nested': True }}",
    }

    # Step 1: Safe local rendering
    rendered = render_template(template, variables)
    print("\n--- Rendered Prompt ---")
    print(rendered)

    # Step 2: Offline model dispatch
    adapter = OfflineModelAdapter()
    req = ModelRequest(prompt=rendered, model="offline-gpt-4-turbo", temperature=0.2, max_tokens=100)
    response = adapter.dispatch(req)

    print("\n--- Model Response ---")
    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Output: {response.output_text}")
    print(f"Usage: {response.usage}")
    print("\nCompleted successfully without remote network access.")


if __name__ == "__main__":
    main()
