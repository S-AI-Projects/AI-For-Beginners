"""Unit tests for Bounding-Box Intersection over Union (R38-T050).

Tests core requirements, boundary conditions, edge/corner touching,
inverted coordinates, mathematical properties, and input validation.
"""

import math
import unittest

from example import (
    box_area,
    compute_iou_summary,
    intersection_area,
    intersection_over_union,
    validate_box,
)


class TestCoreIoURequirements(unittest.TestCase):
    """Verify the mandatory task fixtures and core IoU formulas."""

    def test_partial_overlap_fixture(self):
        """Boxes [0, 0, 2, 2] and [1, 1, 3, 3] have IoU = 1/7."""
        box_a = [0, 0, 2, 2]
        box_b = [1, 1, 3, 3]
        iou = intersection_over_union(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=7)

    def test_identical_boxes_fixture(self):
        """Identical non-degenerate boxes [0, 0, 2, 2] have IoU = 1.0."""
        box = [0, 0, 2, 2]
        iou = intersection_over_union(box, box)
        self.assertEqual(iou, 1.0)

    def test_disjoint_boxes_fixture(self):
        """Disjoint boxes [0, 0, 1, 1] and [2, 2, 3, 3] have IoU = 0.0."""
        box_a = [0, 0, 1, 1]
        box_b = [2, 2, 3, 3]
        iou = intersection_over_union(box_a, box_b)
        self.assertEqual(iou, 0.0)


class TestBoundaryCases(unittest.TestCase):
    """Verify boundary cases: edge touching, corner touching, contained, and degenerate boxes."""

    def test_horizontal_edge_touching(self):
        """Boxes touching along a vertical edge have 0 intersection and 0.0 IoU."""
        box_a = [0, 0, 1, 1]
        box_b = [1, 0, 2, 1]
        self.assertEqual(intersection_area(box_a, box_b), 0.0)
        self.assertEqual(intersection_over_union(box_a, box_b), 0.0)

    def test_vertical_edge_touching(self):
        """Boxes touching along a horizontal edge have 0 intersection and 0.0 IoU."""
        box_a = [0, 0, 1, 1]
        box_b = [0, 1, 1, 2]
        self.assertEqual(intersection_area(box_a, box_b), 0.0)
        self.assertEqual(intersection_over_union(box_a, box_b), 0.0)

    def test_corner_touching(self):
        """Boxes touching at a single corner vertex have 0 intersection and 0.0 IoU."""
        box_a = [0, 0, 1, 1]
        box_b = [1, 1, 2, 2]
        self.assertEqual(intersection_area(box_a, box_b), 0.0)
        self.assertEqual(intersection_over_union(box_a, box_b), 0.0)

    def test_contained_box(self):
        """Box inside another: area_a=16, area_b=4, inter=4, union=16 -> IoU=0.25."""
        outer = [0, 0, 4, 4]
        inner = [1, 1, 3, 3]
        iou = intersection_over_union(outer, inner)
        self.assertAlmostEqual(iou, 0.25, places=7)

    def test_zero_area_point_box(self):
        """Degenerate zero-area point box has union = 0.0 and safely returns 0.0 IoU."""
        point_box = [1, 1, 1, 1]
        iou = intersection_over_union(point_box, point_box)
        self.assertEqual(iou, 0.0)

    def test_zero_area_line_box(self):
        """Degenerate zero-width or zero-height line box has 0 area."""
        line_box = [0, 0, 0, 2]
        self.assertEqual(box_area(line_box), 0.0)
        self.assertEqual(intersection_over_union(line_box, line_box), 0.0)

    def test_point_box_inside_larger_box(self):
        """Zero-area point box inside an area box yields 0 intersection and 0.0 IoU."""
        box = [0, 0, 2, 2]
        point = [1, 1, 1, 1]
        self.assertEqual(intersection_area(box, point), 0.0)
        self.assertEqual(intersection_over_union(box, point), 0.0)

    def test_negative_coordinates(self):
        """Boxes with negative coordinates calculate valid areas and overlap."""
        box_a = [-2, -2, 0, 0]
        box_b = [-1, -1, 1, 1]
        iou = intersection_over_union(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=7)

    def test_floating_point_coordinates(self):
        """Boxes with floating-point values compute accurate IoU."""
        box_a = [0.5, 0.5, 2.5, 2.5]
        box_b = [1.5, 1.5, 3.5, 3.5]
        iou = intersection_over_union(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=7)


class TestMathematicalProperties(unittest.TestCase):
    """Verify symmetry and mathematical range invariants of IoU."""

    def test_symmetry(self):
        """IoU(A, B) must equal IoU(B, A) for any pair of valid boxes."""
        pairs = [
            ([0, 0, 2, 2], [1, 1, 3, 3]),
            ([0, 0, 1, 1], [2, 2, 3, 3]),
            ([0, 0, 4, 4], [1, 1, 3, 3]),
            ([1, 2, 5, 8], [2, 3, 6, 9]),
            ([-3, -5, -1, -2], [-2, -4, 0, 0]),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(
                    intersection_over_union(a, b),
                    intersection_over_union(b, a),
                    places=7,
                )

    def test_iou_range_invariants(self):
        """IoU must always be bounded within [0.0, 1.0]."""
        boxes = [
            [0, 0, 2, 2],
            [1, 1, 3, 3],
            [0, 0, 1, 1],
            [5, 5, 10, 10],
            [-2, -2, 0, 0],
            [0, 0, 100, 100],
        ]
        for i, a in enumerate(boxes):
            for b in boxes[i:]:
                iou = intersection_over_union(a, b)
                self.assertGreaterEqual(iou, 0.0)
                self.assertLessEqual(iou, 1.0)


class TestInputValidationAndErrorHandling(unittest.TestCase):
    """Verify robust validation of coordinate conventions and boundary constraints."""

    def test_inverted_x_raises_value_error(self):
        """x_min > x_max must raise ValueError."""
        inverted_x = [3, 0, 1, 2]
        valid_box = [0, 0, 2, 2]
        with self.assertRaises(ValueError) as ctx:
            intersection_over_union(inverted_x, valid_box)
        self.assertIn("inverted x", str(ctx.exception))

    def test_inverted_y_raises_value_error(self):
        """y_min > y_max must raise ValueError."""
        inverted_y = [0, 5, 2, 1]
        valid_box = [0, 0, 2, 2]
        with self.assertRaises(ValueError) as ctx:
            intersection_over_union(valid_box, inverted_y)
        self.assertIn("inverted y", str(ctx.exception))

    def test_wrong_coordinate_count_raises_value_error(self):
        """Boxes with fewer or more than 4 coordinates must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_box([0, 0, 1])
        with self.assertRaises(ValueError):
            validate_box([0, 0, 1, 1, 1])

    def test_boolean_coordinate_raises_type_error(self):
        """Boolean coordinates must not be silently cast to 0/1 integers."""
        with self.assertRaises(TypeError):
            validate_box([0, True, 2, 2])
        with self.assertRaises(TypeError):
            validate_box([False, 0, 2, 2])

    def test_non_numeric_coordinate_raises_type_error(self):
        """String or non-numeric types must raise TypeError."""
        with self.assertRaises(TypeError):
            validate_box(["0", 0, 2, 2])

    def test_non_finite_coordinates_raise_value_error(self):
        """NaN and inf values must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_box([0, float("nan"), 2, 2])
        with self.assertRaises(ValueError):
            validate_box([0, 0, float("inf"), 2])

    def test_non_sequence_raises_type_error(self):
        """Non-sequence box inputs must raise TypeError."""
        with self.assertRaises(TypeError):
            validate_box(1234)
        with self.assertRaises(TypeError):
            validate_box(None)


class TestHelperFunctions(unittest.TestCase):
    """Verify auxiliary helper functions."""

    def test_validate_box_output_types(self):
        """validate_box returns tuple of floats."""
        coords = validate_box([0, 1, 2, 3])
        self.assertEqual(coords, (0.0, 1.0, 2.0, 3.0))
        self.assertTrue(all(isinstance(c, float) for c in coords))

    def test_box_area_calculation(self):
        """box_area computes width * height correctly."""
        self.assertEqual(box_area([0, 0, 3, 4]), 12.0)

    def test_compute_iou_summary_keys(self):
        """compute_iou_summary returns expected dict structure."""
        summary = compute_iou_summary([0, 0, 2, 2], [1, 1, 3, 3])
        self.assertIn("box_a", summary)
        self.assertIn("box_b", summary)
        self.assertAlmostEqual(summary["area_a"], 4.0)
        self.assertAlmostEqual(summary["area_b"], 4.0)
        self.assertAlmostEqual(summary["intersection"], 1.0)
        self.assertAlmostEqual(summary["union"], 7.0)
        self.assertAlmostEqual(summary["iou"], 1.0 / 7.0)


if __name__ == "__main__":
    unittest.main()

