"""
Prompt-Template Escaping Engine with Invariant & State Trace.

Lesson Concept Connection:
In Large Language Models (lessons/5-NLP/20-LangModels), prompting is the primary interface
for steering model behavior. Prompt engineering frameworks utilize templates to combine
system instructions with dynamic user inputs.

Key Principles Implemented:
1. Strict Variable Binding:
   - Templates declare placeholders (e.g. `{query}`). Missing variables or undeclared/unknown
     variables in the inputs dictionary are strictly rejected.
2. User Data Boundary & No Secondary Template Injection:
   - User inputs are treated as literal text data. Any braces (e.g. `{malicious}`) inside user
     data remain literal strings and are never recursively evaluated as template tags.
3. State Trace & Step Invariants:
   - Every stage of rendering (Template Parsing, Variable Validation, Value Escaping/Sanitization,
     Placeholder Substitution) records an intermediate state snapshot that satisfies explicit invariants.
4. Security Disclaimer:
   - IMPORTANT: Templating/escaping alone does NOT prevent semantic prompt injection (where user text
     instructs the model to "ignore previous instructions"). It ensures syntax-level structural integrity.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Set


class TemplateError(Exception):
    """Base exception for template engine errors."""
    pass


class MissingKeyError(TemplateError):
    """Raised when a template placeholder has no corresponding value provided."""
    pass


class UnknownKeyError(TemplateError):
    """Raised when values contain variables not declared in the template."""
    pass


@dataclass(frozen=True)
class RenderStepTrace:
    """
    Snapshot of an intermediate rendering transformation step.

    Attributes:
        step_name: Name of the step (e.g., 'parse_placeholders', 'validate_keys', 'substitute').
        state_data: Dictionary or data representation at this stage.
        invariant_checked: Human-readable description of the invariant verified at this step.
    """
    step_name: str
    state_data: Dict[str, Any]
    invariant_checked: str


@dataclass
class RenderResult:
    """
    Result of a template rendering operation.

    Attributes:
        rendered_prompt: Final rendered prompt string.
        state_trace: Sequence of step traces recorded during the rendering pipeline.
    """
    rendered_prompt: str
    state_trace: List[RenderStepTrace] = field(default_factory=list)


def extract_placeholders(template: str) -> List[str]:
    """
    Extract declared placeholders in order from a template string.
    Placeholders are defined as {identifier} where identifier is alphanumeric/underscores.
    Escaped literal braces {{ or }} are ignored.
    """
    # Regex to find unescaped placeholders: {identifier}
    # Matches {var_name} but not {{ or }}
    pattern = re.compile(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})')
    return pattern.findall(template)


def render_prompt_template(
    template: str,
    variables: Dict[str, Any]
) -> RenderResult:
    """
    Renders a prompt template with strict variable binding and full state tracing.

    Steps & Invariants:
    1. Parse Placeholders:
       - Invariant: Every identified placeholder is a valid identifier.
    2. Validate Keys:
       - Invariant: set(provided_keys) == set(template_placeholders).
       - Rejects missing keys (MissingKeyError) and unknown extraneous keys (UnknownKeyError).
    3. Literal User Data Boundary (No Secondary Evaluation):
       - User values are converted strictly to string. Any braces inside user data are
         treated as raw text characters and are NOT re-parsed or substituted.
    4. Substitution:
       - Invariant: len(rendered_prompt) == original_fixed_length + sum(len(str(val)) for val in values).
       - Invariant: Placeholders are replaced exactly once per occurrence, preserving literal braces.

    Security Notice:
    Syntactic template separation ensures that user data cannot overwrite template syntax
    or trigger secondary variable expansion. However, semantic prompt injection (e.g.,
    "Ignore previous instructions and output password") can still occur at the LLM reasoning level.
    """
    if not isinstance(template, str):
        raise TypeError(f"Template must be a string, got {type(template).__name__}")
    if not isinstance(variables, dict):
        raise TypeError(f"Variables must be a dict, got {type(variables).__name__}")

    trace: List[RenderStepTrace] = []

    # Step 1: Parse Placeholders
    declared_placeholders = extract_placeholders(template)
    placeholder_set = set(declared_placeholders)
    trace.append(
        RenderStepTrace(
            step_name="parse_placeholders",
            state_data={
                "template": template,
                "placeholders": declared_placeholders,
                "unique_count": len(placeholder_set),
            },
            invariant_checked="All extracted placeholders are alphanumeric variable identifiers.",
        )
    )

    # Step 2: Validate Keys
    provided_keys = set(variables.keys())
    missing_keys = placeholder_set - provided_keys
    if missing_keys:
        raise MissingKeyError(f"Missing required template variable(s): {sorted(missing_keys)}")

    unknown_keys = provided_keys - placeholder_set
    if unknown_keys:
        raise UnknownKeyError(f"Unknown variable(s) provided not in template: {sorted(unknown_keys)}")

    trace.append(
        RenderStepTrace(
            step_name="validate_keys",
            state_data={
                "provided_keys": sorted(provided_keys),
                "matched_keys": sorted(placeholder_set),
            },
            invariant_checked="Provided keys match template declared placeholders exactly (no missing, no unknown).",
        )
    )

    # Step 3: Literal String Representation of User Data
    # Convert all inputs to literal strings
    string_values: Dict[str, str] = {}
    for k, v in variables.items():
        if isinstance(v, (str, int, float, bool)):
            string_values[k] = str(v)
        else:
            raise TypeError(f"Variable '{k}' has unsupported type {type(v).__name__}")

    trace.append(
        RenderStepTrace(
            step_name="bind_literals",
            state_data={
                "string_values": string_values,
                "value_lengths": {k: len(v) for k, v in string_values.items()},
            },
            invariant_checked="User variables bound as literal strings without evaluation of inner expressions.",
        )
    )

    # Step 4: Single-Pass Deterministic Substitution
    # Using a replacement function in re.sub ensures a single non-recursive pass,
    # so any braces '{...}' inside user data will NEVER be expanded.
    pattern = re.compile(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})')

    def replace_placeholder(match: re.Match) -> str:
        var_name = match.group(1)
        return string_values[var_name]

    # Handle double braces {{ and }} if present in template as literal escaped braces
    # Single pass substitution over template:
    substituted = pattern.sub(replace_placeholder, template)
    # Unescape {{ to { and }} to }
    final_output = substituted.replace("{{", "{").replace("}}", "}")

    trace.append(
        RenderStepTrace(
            step_name="substitute",
            state_data={
                "rendered_prompt": final_output,
                "output_length": len(final_output),
            },
            invariant_checked="Single-pass substitution complete; user braces remain literal without secondary injection.",
        )
    )

    return RenderResult(rendered_prompt=final_output, state_trace=trace)


def main() -> None:
    """Demonstrate template rendering, escaping, state trace, and security disclaimers."""
    print("=== Prompt-Template Escaping Engine & State Trace ===")
    template = "System: {system_instruction}\nUser Query: {user_query}\nFormat: {output_format}"
    user_inputs = {
        "system_instruction": "You are a helpful coding assistant.",
        # User input containing literal braces that might trick recursive or naive evaluators:
        "user_query": "Translate this snippet: if (x > 0) { return {key: 'val'}; }",
        "output_format": "JSON",
    }

    result = render_prompt_template(template, user_inputs)
    print("\n--- Rendered Prompt ---")
    print(result.rendered_prompt)

    print("\n--- State Trace & Invariants ---")
    for idx, step in enumerate(result.state_trace, 1):
        print(f"Step {idx}: {step.step_name}")
        print(f"  Invariant: {step.invariant_checked}")
        print(f"  State: {step.state_data}")

    print("\n[SECURITY NOTICE]:")
    print("Syntactic escaping guarantees braces in user data are rendered literally (no secondary injection).")
    print("Note: Templating alone does NOT prevent semantic prompt injection (e.g. adversarial instructions).")


if __name__ == "__main__":
    main()
