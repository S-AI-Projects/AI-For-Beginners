"""
Tensor-Shape Contract — Invariant and State Trace
Applied lab for lessons/4-ComputerVision/10-GANs (Task R38-T014)

Concept & Contract:
In GAN training pipelines (e.g. DCGANs), image generators and discriminators
require strict shape consistency across batch transformations:
- Image batch input: N x H x W numeric pixels in range [0, 255].
- Normalization: Scale [0, 255] -> [0.0, 1.0] (0 -> 0.0, 255 -> 1.0).
- Shape Preservation: Input batch [[[0, 255], [128, 64]]] maintains shape (1, 2, 2).
- Validation: Negative values (< 0) or values exceeding 255 (> 255) must be rejected.
- Pedagogical Focus: Record intermediate states after each pipeline step (Validation,
  Shape Recording, Normalization) and verify invariants linking intermediate states
  to final outputs.
"""

from typing import Any, Dict, List, Tuple, Union

Number = Union[int, float]
Image2D = List[List[Number]]
Batch3D = List[Image2D]
Shape = Tuple[int, int, int]


def get_shape(batch: Batch3D) -> Shape:
    """Determine (N, H, W) shape and check rectangular consistency."""
    if not isinstance(batch, list) or len(batch) == 0:
        raise ValueError("Batch must be a non-empty list.")

    n = len(batch)
    if not isinstance(batch[0], list) or len(batch[0]) == 0:
        raise ValueError("Images in batch must be non-empty lists.")
    h = len(batch[0])

    if not isinstance(batch[0][0], list) or len(batch[0][0]) == 0:
        raise ValueError("Rows in image must be non-empty lists.")
    w = len(batch[0][0])

    for img in batch:
        if not isinstance(img, list) or len(img) != h:
            raise ValueError("All images in batch must have identical height H.")
        for row in img:
            if not isinstance(row, list) or len(row) != w:
                raise ValueError("All rows in images must have identical width W.")

    return (n, h, w)


class PreprocessingStateTracer:
    """Pipelines image normalization while recording state traces and invariants."""

    def __init__(self):
        self.trace: List[Dict[str, Any]] = []

    def validate_and_normalize(self, batch: Batch3D) -> Tuple[Batch3D, List[Dict[str, Any]]]:
        """
        Validate pixels [0, 255], record intermediate states, and scale to [0.0, 1.0].
        """
        self.trace = []

        # Step 1: Shape Inspection
        shape = get_shape(batch)
        self.trace.append({
            "step": "shape_inspection",
            "batch_shape": shape,
            "status": "VALID_STRUCTURE"
        })

        # Step 2: Pixel Range Validation
        for img_idx, img in enumerate(batch):
            for r_idx, row in enumerate(img):
                for c_idx, val in enumerate(row):
                    if not isinstance(val, (int, float)):
                        raise TypeError(f"Pixel value at ({img_idx}, {r_idx}, {c_idx}) must be numeric, got {type(val)}")
                    if val < 0 or val > 255:
                        raise ValueError(f"Pixel value {val} at ({img_idx}, {r_idx}, {c_idx}) out of bounds [0, 255]")

        self.trace.append({
            "step": "pixel_validation",
            "min_bound": 0,
            "max_bound": 255,
            "status": "ALL_PIXELS_IN_RANGE"
        })

        # Step 3: Normalization & Shape Invariant Preservation
        normalized_batch: Batch3D = []
        for img in batch:
            norm_img: Image2D = []
            for row in img:
                norm_img.append([float(val) / 255.0 for val in row])
            normalized_batch.append(norm_img)

        norm_shape = get_shape(normalized_batch)
        invariant_satisfied = (shape == norm_shape)

        self.trace.append({
            "step": "normalization",
            "output_shape": norm_shape,
            "shape_invariant_preserved": invariant_satisfied,
            "status": "COMPLETE"
        })

        if not invariant_satisfied:
            raise RuntimeError("Shape invariant violated during normalization.")

        return normalized_batch, self.trace


def run_demonstration():
    contract_batch: Batch3D = [[[0, 255], [128, 64]]]
    tracer = PreprocessingStateTracer()
    normalized, trace = tracer.validate_and_normalize(contract_batch)

    print("=== Tensor-Shape Contract & State Trace Demo (R38-T014) ===")
    print(f"Input Batch: {contract_batch}")
    print(f"Input Shape: {get_shape(contract_batch)}")
    print(f"Normalized Output: {normalized}")
    print(f"Output Shape: {get_shape(normalized)}")
    print("State Trace:")
    for step in trace:
        print(f"  - {step}")


if __name__ == "__main__":
    run_demonstration()
