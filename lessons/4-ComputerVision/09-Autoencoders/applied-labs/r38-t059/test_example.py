import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import normalize_batch, inspect_batch_shape, run_experiment

class TestTensorShapeContract(unittest.TestCase):
    def test_stated_scenario(self):
        # Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1; negative or >255 rejected
        batch = [[[0, 255], [128, 64]]]
        norm = normalize_batch(batch)
        self.assertEqual(inspect_batch_shape(norm), (1, 2, 2))
        self.assertAlmostEqual(norm[0][0][0], 0.0)
        self.assertAlmostEqual(norm[0][0][1], 1.0)
        self.assertAlmostEqual(norm[0][1][0], 128 / 255.0)
        self.assertAlmostEqual(norm[0][1][1], 64 / 255.0)

    def test_out_of_bounds_rejected(self):
        # Negative pixel rejected
        with self.assertRaises(ValueError):
            normalize_batch([[[-1, 255], [128, 64]]])
        # Pixel > 255 rejected
        with self.assertRaises(ValueError):
            normalize_batch([[[256, 255], [128, 64]]])

    def test_shape_inconsistency_rejected(self):
        # Ragged rows rejected
        with self.assertRaises(ValueError):
            normalize_batch([[[0, 255], [128]]])

    def test_experiment_execution(self):
        exp = run_experiment()
        self.assertIn("1x2x2", exp)
        self.assertEqual(exp["1x2x2"]["shape"], (1, 2, 2))
        self.assertEqual(exp["1x2x2"]["total_pixels"], 4)

    def test_deliberate_failure_check(self):
        batch = [[[0, 255], [128, 64]]]
        norm = normalize_batch(batch)
        self.assertNotEqual(norm[0][0][1], 255.0)

if __name__ == "__main__":
    unittest.main()
