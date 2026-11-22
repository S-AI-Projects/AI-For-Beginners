"""
Tensor-Shape Contract — Evaluation Experiment
Applied lab for lessons/4-ComputerVision/06-IntroCV (Task R38-T019)

Concept & Contract:
In Computer Vision preprocessing pipelines, image batches represented as N x H x W
must be validated and normalized from integer intensities [0, 255] to [0.0, 1.0]:
- Shape Preservation: Batch [[[0, 255], [128, 64]]] remains shape (1, 2, 2).
- Normalized Values: 0 -> 0.0, 255 -> 1.0.
- Rejection: Negative pixels (< 0) or values > 255 raise ValueError.

Focus:
Run an evaluation experiment comparing three small controlled configurations / input sizes:
- Config A: (1, 2, 2) - Contract baseline (4 pixels)
- Config B: (2, 2, 2) - Batch scaling (8 pixels)
- Config C: (1, 4, 4) - Spatial resolution scaling (16 pixels)
Evaluate correctness, throughput/latency scaling, and shape-preservation invariants
without making unsupported claims about real-world trained model accuracy.
"""

import time
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


def run_evaluation_experiment() -> List[Dict[str, Any]]:
    """
    Compare three small controlled configurations:
    Config A: (1, 2, 2)
    Config B: (2, 2, 2)
    Config C: (1, 4, 4)
    """
    configs = {
        "Config A (1x2x2)": [[[0, 255], [128, 64]]],
        "Config B (2x2x2)": [[[0, 255], [128, 64]], [[255, 0], [64, 128]]],
        "Config C (1x4x4)": [
            [
                [0, 64, 128, 255],
                [255, 128, 64, 0],
                [32, 64, 128, 160],
                [200, 220, 240, 255],
            ]
        ],
    }

    experiment_reports = []
    for name, batch in configs.items():
        shape_in = get_shape(batch)
        pixel_count = shape_in[0] * shape_in[1] * shape_in[2]

        # Benchmarking iterations
        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            out = normalize_image_batch(batch)
        elapsed = time.perf_counter() - start

        shape_out = get_shape(out)
        avg_us = (elapsed / iterations) * 1e6

        report = {
            "config_name": name,
            "input_shape": shape_in,
            "output_shape": shape_out,
            "pixel_count": pixel_count,
            "shape_preserved": (shape_in == shape_out),
            "avg_latency_us": round(avg_us, 2),
            "normalized_min": min(p for img in out for row in img for p in row),
            "normalized_max": max(p for img in out for row in img for p in row),
        }
        experiment_reports.append(report)
    return experiment_reports


def run_demonstration():
    contract_batch: Batch3D = [[[0, 255], [128, 64]]]
    normalized = normalize_image_batch(contract_batch)

    print("=== Tensor-Shape Contract: Evaluation Experiment (R38-T019) ===")
    print(f"Contract Input: {contract_batch} -> Shape: {get_shape(contract_batch)}")
    print(f"Contract Output: {normalized} -> Shape: {get_shape(normalized)}")
    print(f"Contract values: 0 -> {normalized[0][0][0]}, 255 -> {normalized[0][0][1]}")
    print("\n--- Comparative Evaluation Experiment Results ---")
    for rep in run_evaluation_experiment():
        print(f"[{rep['config_name']}] Pixels: {rep['pixel_count']}, Shape Invariant: {rep['shape_preserved']}, Avg Latency: {rep['avg_latency_us']} us")


if __name__ == "__main__":
    run_demonstration()
