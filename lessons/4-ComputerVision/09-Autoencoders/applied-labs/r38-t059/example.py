"""
R38-T059: Tensor-shape contract — evaluation experiment
Extend “Autoencoders” at lessons/4-ComputerVision/09-Autoencoders/README.md.
Validate and normalize a small synthetic image batch: accept N×H×W numeric pixels in 0..255 and produce values in 0..1 without changing the batch shape.
Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1; negative or >255 pixels are rejected.
"""
import time
from typing import List, Tuple, Any, Dict

def inspect_batch_shape(batch: Any) -> Tuple[int, int, int]:
    """
    Validates that input is a 3D batch of shape (N, H, W).
    """
    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        raise ValueError("Batch must be a non-empty sequence of images")

    N = len(batch)
    expected_H = None
    expected_W = None

    for i, img in enumerate(batch):
        if not isinstance(img, (list, tuple)) or len(img) == 0:
            raise ValueError(f"Image {i} must be a non-empty 2D sequence")
        H = len(img)
        if expected_H is None:
            expected_H = H
        elif H != expected_H:
            raise ValueError(f"Inconsistent height at image {i}: {H} vs {expected_H}")

        for r, row in enumerate(img):
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise ValueError(f"Row {r} in image {i} must be non-empty sequence")
            W = len(row)
            if expected_W is None:
                expected_W = W
            elif W != expected_W:
                raise ValueError(f"Inconsistent width at image {i}, row {r}: {W} vs {expected_W}")

    return (N, expected_H, expected_W)

def normalize_batch(batch: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Normalize pixel values in [0, 255] to [0.0, 1.0] while preserving (N, H, W) shape.
    Rejects negative pixels or pixels > 255.
    """
    N, H, W = inspect_batch_shape(batch)
    normalized: List[List[List[float]]] = []

    for i, img in enumerate(batch):
        norm_img: List[List[float]] = []
        for r, row in enumerate(img):
            norm_row: List[float] = []
            for c, val in enumerate(row):
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(f"Invalid non-numeric pixel at [{i}][{r}][{c}]: {val}")
                if val < 0.0 or val > 255.0:
                    raise ValueError(f"Pixel value {val} at [{i}][{r}][{c}] out of bounds [0, 255]")
                norm_row.append(val / 255.0)
            norm_img.append(norm_row)
        normalized.append(norm_img)

    return normalized

def run_experiment() -> Dict[str, Any]:
    """
    Controlled evaluation experiment comparing 3 small input configurations.
    Reports element count and processing correctness without claiming model accuracy.
    """
    configs = {
        "1x2x2": [[[0, 255], [128, 64]]],
        "2x2x2": [[[0, 255], [128, 64]], [[255, 0], [64, 128]]],
        "1x3x3": [[[0, 128, 255], [64, 192, 32], [255, 0, 128]]]
    }
    results = {}
    for name, b in configs.items():
        norm = normalize_batch(b)
        shape = inspect_batch_shape(norm)
        num_pixels = shape[0] * shape[1] * shape[2]
        results[name] = {"shape": shape, "total_pixels": num_pixels, "valid": True}
    return results

def main():
    b = [[[0, 255], [128, 64]]]
    norm = normalize_batch(b)
    print("Normalized batch:", norm)
    print("Experiment:", run_experiment())

if __name__ == "__main__":
    main()
