"""Token-budget preflight for lesson 19 (Named Entity Recognition).

TEACHING APPROXIMATION: tokens are whitespace-separated words. A real model
uses its own subword tokenizer, so real counts will differ. This file makes no
claim about real hardware, cloud services or trained-model performance.
"""
import sys

TOKENIZER_LABEL = "whitespace (teaching approximation, not the model's real tokenizer)"


def tokenize(text):
    return text.split()


def preflight(text, budget):
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must not be negative")

    trace = []  # intermediate state recorded after each step
    tokens = tokenize(text)
    trace.append({"step": "tokenize", "tokens": list(tokens)})
    retained = tokens[:budget]
    trace.append({"step": "retain", "retained": list(retained)})
    removed = tokens[budget:]
    trace.append({"step": "report", "removed": list(removed)})

    return {
        "budget": budget,
        "tokenizer": TOKENIZER_LABEL,
        "retained": retained,
        "removed": removed,
        "trace": trace,
    }


def check_invariant(result):
    """Invariant linking the recorded state to the final output:
    retained + removed == original tokens, in order, and retained fits the budget."""
    tokens = result["trace"][0]["tokens"]
    retained_state = result["trace"][1]["retained"]
    removed_state = result["trace"][2]["removed"]
    checks = [
        (retained_state + removed_state == tokens, "state does not rebuild the input"),
        (result["retained"] == retained_state, "final retained differs from traced state"),
        (result["removed"] == removed_state, "final removed differs from traced state"),
        (len(result["retained"]) <= result["budget"], "retained exceeds budget"),
        (len(result["retained"]) == min(len(tokens), result["budget"]), "wrong retained count"),
    ]
    for ok, message in checks:
        if not ok:
            raise AssertionError(message)


def render(result):
    removed = result["removed"]
    return "\n".join([
        f"Tokenizer: {result['tokenizer']}",
        f"Budget: {result['budget']}",
        f"Retained: [{','.join(result['retained'])}]",
        f"Removed: {len(removed)} token(s) [{','.join(removed)}]",
    ])


def main(argv):
    if len(argv) != 3:
        print('usage: example.py "<text>" <budget>', file=sys.stderr)
        return 2
    try:
        result = preflight(argv[1], int(argv[2]))
    except (ValueError, TypeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    check_invariant(result)
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
