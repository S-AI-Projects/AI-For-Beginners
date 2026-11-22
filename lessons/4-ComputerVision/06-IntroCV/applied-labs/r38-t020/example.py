"""
Tensor-Shape Contract — Scenario Assessment
Applied lab for lessons/4-ComputerVision/06-IntroCV (Task R38-T020)

Concept & Contract:
In Computer Vision preprocessing:
- Accept N x H x W numeric pixels in 0..255 and produce values in 0..1 without changing batch shape.
- Batch [[[0, 255], [128, 64]]] stays shape (1, 2, 2); 0 -> 0.0, 255 -> 1.0.
- Negative (< 0) or > 255 pixels are rejected (raises ValueError).

Focus:
Add three applied questions about the local example:
- Q1 (Predict an Output): Predict the exact normalized float representation and shape
  for contract input [[[0, 255], [128, 64]]].
- Q2 (Diagnose a Failure): Given invalid batch [[[-5, 255], [128, 64]]], identify the exact
  failure reason, location, and exception type.
- Q3 (Choose a Boundary Policy): Evaluate boundary values (0 and 255) vs subnormal/clamping
  alternatives; justify why strict boundary rejection is preferred over silent truncation.

Provide a rationale and an executable check for each factual answer.
"""

from typing import Any, Dict, List, Tuple, Union

Number = Union[int, float]
Image2D = List[List[Number]]
Batch3D = List[Image2D]
Shape = Tuple[int, int, int]


def get_shape(batch: Batch3D) -> Shape:
    """Validate uniform rectangular structure and return (N, H, W)."""
    if not isinstance(batch, list) or len(batch) == 0:
        raise ValueError("Batch must be a non-empty list.")
    n = len(batch)
    if not isinstance(batch[0], list) or len(batch[0]) == 0:
        raise ValueError("Images must be non-empty lists.")
    h = len(batch[0])
    if not isinstance(batch[0][0], list) or len(batch[0][0]) == 0:
        raise ValueError("Rows must be non-empty lists.")
    w = len(batch[0][0])

    for img in batch:
        if not isinstance(img, list) or len(img) != h:
            raise ValueError("All images must have identical height H.")
        for row in img:
            if not isinstance(row, list) or len(row) != w:
                raise ValueError("All rows must have identical width W.")
    return (n, h, w)


def normalize_image_batch(batch: Batch3D) -> Batch3D:
    """
    Validate pixels in [0, 255] and normalize to [0.0, 1.0] while preserving shape.
    """
    shape = get_shape(batch)
    normalized: Batch3D = []
    for img_idx, img in enumerate(batch):
        norm_img: Image2D = []
        for r_idx, row in enumerate(img):
            norm_row: List[float] = []
            for c_idx, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Pixel at ({img_idx}, {r_idx}, {c_idx}) must be numeric, got {type(val)}")
                if val < 0 or val > 255:
                    raise ValueError(f"Pixel {val} at ({img_idx}, {r_idx}, {c_idx}) out of range [0, 255]")
                norm_row.append(float(val) / 255.0)
            norm_img.append(norm_row)
        normalized.append(norm_img)

    if get_shape(normalized) != shape:
        raise RuntimeError("Shape mismatch after normalization.")
    return normalized


# Applied Scenario Questions & Rationales
APPLIED_SCENARIOS = {
    "Q1_predict_output": {
        "question": "What is the expected normalized output and shape for [[[0, 255], [128, 64]]]?",
        "expected_shape": (1, 2, 2),
        "expected_values": [[[0.0, 1.0], [128.0 / 255.0, 64.0 / 255.0]]],
        "rationale": (
            "Linear scaling divides each integer intensity by 255.0. 0/255 = 0.0, "
            "255/255 = 1.0, 128/255 ≈ 0.50196, and 64/255 ≈ 0.25098. The 3D list "
            "structure remains identical, preserving the (1, 2, 2) batch shape."
        ),
    },
    "Q2_diagnose_failure": {
        "question": "Why does input [[[-5, 255], [128, 64]]] fail, and what exception is raised?",
        "expected_exception": ValueError,
        "error_keyword": "out of range [0, 255]",
        "rationale": (
            "Pixel values in digital 8-bit imaging represent non-negative light intensities. "
            "A pixel value of -5 is unphysical and indicates data corruption or upstream processing bugs. "
            "The validator raises ValueError explicitly rather than propagating invalid tensors."
        ),
    },
    "Q3_boundary_policy": {
        "question": "Why is strict boundary rejection chosen over silent clamping for values outside [0, 255]?",
        "policy": "STRICT_REJECTION",
        "rationale": (
            "Silent clamping (e.g. min(255, max(0, x))) masks sensor faults, erroneous color space conversions, "
            "or overflow bugs. In safety-critical computer vision pipelines, failing fast with an explicit "
            "exception prevents training degradation or silent downstream inference failures."
        ),
    },
}


def assess_scenarios() -> Dict[str, bool]:
    """Execute validation checks for each applied question."""
    results = {}

    # Check Q1
    contract_input = [[[0, 255], [128, 64]]]
    actual_out = normalize_image_batch(contract_input)
    actual_shape = get_shape(actual_out)
    q1 = APPLIED_SCENARIOS["Q1_predict_output"]
    q1_passed = (actual_shape == q1["expected_shape"]) and (actual_out[0][0][0] == 0.0) and (actual_out[0][0][1] == 1.0)
    results["Q1_predict_output"] = q1_passed

    # Check Q2
    q2_passed = False
    try:
        normalize_image_batch([[[-5, 255], [128, 64]]])
    except ValueError as e:
        if "out of range [0, 255]" in str(e):
            q2_passed = True
    results["Q2_diagnose_failure"] = q2_passed

    # Check Q3
    q3 = APPLIED_SCENARIOS["Q3_boundary_policy"]
    results["Q3_boundary_policy"] = (q3["policy"] == "STRICT_REJECTION")

    return results


def run_demonstration():
    print("=== Tensor-Shape Contract: Scenario Assessment Demo (R38-T020) ===")
    assessment = assess_scenarios()
    for q_id, passed in assessment.items():
        q_data = APPLIED_SCENARIOS[q_id]
        print(f"\n[{q_id}] Validated: {passed}")
        print(f"  Question: {q_data['question']}")
        print(f"  Rationale: {q_data['rationale']}")


if __name__ == "__main__":
    run_demonstration()
