"""Bounding-Box Intersection over Union (IoU) with Boundary-Case Handling.

Applied Lab: R38-T050
Lesson: lessons/4-ComputerVision/11-ObjectDetection/

TEACHING LAB: This module demonstrates the geometric foundation of object
detection evaluation: Intersection over Union (IoU) between axis-aligned
bounding boxes using continuous coordinates [x_min, y_min, x_max, y_max].

Key Concepts:
- Bounding Box: A rectangle enclosing an object, defined by top-left (x_min, y_min)
  and bottom-right (x_max, y_max) corners.
- Coordinate Convention: Continuous geometric coordinates where
  width = x_max - x_min and height = y_max - y_min (no discrete pixel +1 offset).
- Overlap & Intersection: The rectangular region shared by both boxes.
- Union: The combined area covered by either box (Area_A + Area_B - Intersection).
- IoU: Ratio of intersection area to union area: IoU = Intersection / Union.
  Range: [0.0, 1.0].
- Boundary Policies:
  - Touching edges or corners yield 0.0 intersection area and 0.0 IoU.
  - Degenerate zero-area boxes resulting in zero union yield 0.0 IoU (safe sentinel).
  - Inverted coordinates (x_min > x_max or y_min > y_max) raise ValueError.
  - Non-numeric or boolean coordinates raise TypeError.
  - Non-finite coordinates (NaN, inf) raise ValueError.
"""

import math
from typing import Any, Dict, Sequence, Tuple


def validate_box(box: Any, name: str = "box") -> Tuple[float, float, float, float]:
    """Validate and normalize bounding box coordinates [x_min, y_min, x_max, y_max].

    Args:
        box: A sequence containing 4 numerical coordinates.
        name: Name of the box parameter for informative error messages.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) as floats.

    Raises:
        TypeError: If box is not a list/tuple, or contains non-numeric / boolean values.
        ValueError: If box does not have length 4, has non-finite values, or is inverted.
    """
    if not isinstance(box, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of 4 coordinates, got {type(box).__name__}")
    if len(box) != 4:
        raise ValueError(
            f"{name} must contain exactly 4 coordinates [x_min, y_min, x_max, y_max], got {len(box)}"
        )
    for i, coord in enumerate(box):
        if isinstance(coord, bool):
            raise TypeError(f"{name}[{i}] must be int or float, got bool ({coord})")
        if not isinstance(coord, (int, float)):
            raise TypeError(f"{name}[{i}] must be int or float, got {type(coord).__name__}")
        if math.isnan(coord) or math.isinf(coord):
            raise ValueError(f"{name}[{i}] must be a finite real number, got {coord}")

    x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x_min > x_max:
        raise ValueError(f"{name} has inverted x coordinates: x_min ({x_min}) > x_max ({x_max})")
    if y_min > y_max:
        raise ValueError(f"{name} has inverted y coordinates: y_min ({y_min}) > y_max ({y_max})")

    return (x_min, y_min, x_max, y_max)


def box_area(box: Sequence[float], name: str = "box") -> float:
    """Compute the 2D area of an axis-aligned bounding box.

    In continuous coordinates: area = (x_max - x_min) * (y_max - y_min).
    """
    x_min, y_min, x_max, y_max = validate_box(box, name=name)
    return (x_max - x_min) * (y_max - y_min)


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute the area of intersection between two axis-aligned bounding boxes.

    Returns 0.0 if the boxes are disjoint, touch only at an edge, or touch at a corner.
    """
    ax_min, ay_min, ax_max, ay_max = validate_box(box_a, name="box_a")
    bx_min, by_min, bx_max, by_max = validate_box(box_b, name="box_b")

    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)

    inter_width = max(0.0, inter_x_max - inter_x_min)
    inter_height = max(0.0, inter_y_max - inter_y_min)

    return inter_width * inter_height


def intersection_over_union(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    IoU = Intersection(A, B) / Union(A, B)
    where Union(A, B) = Area(A) + Area(B) - Intersection(A, B).

    Boundary policies:
    - If Union == 0.0 (e.g. two degenerate point boxes), returns 0.0.
    - If boxes do not overlap (or only touch edges/corners), returns 0.0.
    - If boxes are identical and non-degenerate, returns 1.0.

    Returns:
        IoU as float in range [0.0, 1.0].
    """
    inter = intersection_area(box_a, box_b)
    area_a = box_area(box_a, name="box_a")
    area_b = box_area(box_b, name="box_b")

    union = area_a + area_b - inter
    if union == 0.0:
        return 0.0

    return inter / union


def compute_iou_summary(box_a: Sequence[float], box_b: Sequence[float]) -> Dict[str, Any]:
    """Compute a detailed summary dictionary of box geometric relationship."""
    ax_min, ay_min, ax_max, ay_max = validate_box(box_a, name="box_a")
    bx_min, by_min, bx_max, by_max = validate_box(box_b, name="box_b")

    area_a = (ax_max - ax_min) * (ay_max - ay_min)
    area_b = (bx_max - bx_min) * (by_max - by_min)

    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter = inter_w * inter_h

    union = area_a + area_b - inter
    iou = (inter / union) if union > 0.0 else 0.0

    return {
        "box_a": [ax_min, ay_min, ax_max, ay_max],
        "box_b": [bx_min, by_min, bx_max, by_max],
        "area_a": area_a,
        "area_b": area_b,
        "intersection": inter,
        "union": union,
        "iou": iou,
    }


def run_demo() -> None:
    """Run standard demonstration scenarios including normal, boundary, and error cases."""
    print("=" * 70)
    print("R38-T050: Bounding-Box Intersection over Union (IoU) Demonstration")
    print("=" * 70)

    fixtures = [
        ("Partial Overlap (1/7)", [0, 0, 2, 2], [1, 1, 3, 3], 1.0 / 7.0),
        ("Identical Boxes (1.0)", [0, 0, 2, 2], [0, 0, 2, 2], 1.0),
        ("Disjoint Boxes (0.0)", [0, 0, 1, 1], [2, 2, 3, 3], 0.0),
        ("Touching Along Edge (0.0)", [0, 0, 1, 1], [1, 0, 2, 1], 0.0),
        ("Touching at Corner (0.0)", [0, 0, 1, 1], [1, 1, 2, 2], 0.0),
        ("Contained Box (0.25)", [0, 0, 4, 4], [1, 1, 3, 3], 0.25),
        ("Negative Coordinates", [-2, -2, 0, 0], [-1, -1, 1, 1], 1.0 / 7.0),
        ("Zero-Area Point Boxes (0.0)", [1, 1, 1, 1], [1, 1, 1, 1], 0.0),
    ]

    print("\n--- Standard & Boundary Scenarios ---")
    for label, b1, b2, expected in fixtures:
        summary = compute_iou_summary(b1, b2)
        print(f"\nScenario: {label}")
        print(f"  Box A: {summary['box_a']} (Area: {summary['area_a']:.2f})")
        print(f"  Box B: {summary['box_b']} (Area: {summary['area_b']:.2f})")
        print(f"  Intersection: {summary['intersection']:.4f}")
        print(f"  Union:        {summary['union']:.4f}")
        print(f"  IoU:          {summary['iou']:.6f} (Expected: {expected:.6f})")

    print("\n--- Error Handling Scenarios ---")
    error_cases = [
        ("Inverted X coordinates", [3, 0, 1, 2], [0, 0, 2, 2]),
        ("Inverted Y coordinates", [0, 5, 2, 1], [0, 0, 2, 2]),
        ("Wrong coordinate count (3 items)", [0, 0, 2], [0, 0, 2, 2]),
        ("Boolean coordinate", [0, True, 2, 2], [0, 0, 2, 2]),
        ("Non-numeric coordinate", ["zero", 0, 2, 2], [0, 0, 2, 2]),
        ("Non-finite NaN coordinate", [0, float("nan"), 2, 2], [0, 0, 2, 2]),
    ]
    for label, b1, b2 in error_cases:
        try:
            intersection_over_union(b1, b2)
            print(f"  [UNEXPECTED SUCCESS] {label}")
        except (ValueError, TypeError) as exc:
            print(f"  [REJECTED AS EXPECTED] {label}: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("Demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()

