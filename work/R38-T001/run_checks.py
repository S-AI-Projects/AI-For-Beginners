#!/usr/bin/env python3
"""
Verification check for R38-T001.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

# Add applied-labs directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/4-ComputerVision/10-GANs/applied-labs/r38-t001")))
from example import validate_and_normalize_batch, get_batch_shape

def main():
    print("Running verification for R38-T001...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    # Record tool versions
    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T001",
        "lesson": "lessons/4-ComputerVision/10-GANs"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    # Canonical fixture
    input_fixture = [[[0, 255], [128, 64]]]
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump({"canonical_batch": input_fixture}, f, indent=2)

    # Actual result
    actual_shape = get_batch_shape(input_fixture)
    actual_normalized = validate_and_normalize_batch(input_fixture)

    expected = {
        "shape": [1, 2, 2],
        "normalized": [[[0.0, 1.0], [128 / 255.0, 64 / 255.0]]]
    }
    actual = {
        "shape": list(actual_shape),
        "normalized": actual_normalized
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    # Check contract assertions
    assert actual["shape"] == expected["shape"], f"Shape mismatch: {actual['shape']} vs {expected['shape']}"
    assert abs(actual_normalized[0][0][0] - 0.0) < 1e-6, "0 must map to 0.0"
    assert abs(actual_normalized[0][0][1] - 1.0) < 1e-6, "255 must map to 1.0"
    
    # Check invalid pixel rejection
    try:
        validate_and_normalize_batch([[[0, -1], [128, 64]]])
        raise AssertionError("Failed to reject negative pixel")
    except ValueError:
        pass

    try:
        validate_and_normalize_batch([[[0, 256], [128, 64]]])
        raise AssertionError("Failed to reject pixel > 255")
    except ValueError:
        pass

    print("All checks passed successfully for R38-T001.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
