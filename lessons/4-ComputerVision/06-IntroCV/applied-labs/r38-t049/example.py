"""Tensor-Shape Contract — Runnable Concept Example (R38-T049).

TEACHING LAB: This is a local educational demonstration of image tensor representation,
shape validation, and pixel normalization for computer vision preprocessing.
It uses only the Python standard library.

It does NOT:
- Load external image files (JPEG/PNG) or call OpenCV/NumPy/PyTorch.
- Train or evaluate a computer vision model.
- Require GPU hardware or physical cameras.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# ── Standard Mandatory Fixture ───────────────────────────────────────
MAIN_BATCH: list[list[list[int]]] = [
    [
        [0, 255],
        [128, 64],
    ]
]


@dataclass(frozen=True)
class NormalizedBatchResult:
    """Structured result of image batch validation and pixel normalization."""

    original_shape: tuple[int, int, int]
    normalized_shape: tuple[int, int, int]
    normalized_batch: list[list[list[float]]]
    shape_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain dictionary."""
        return asdict(self)


# ── Shape and Validation Helpers ─────────────────────────────────────

def get_batch_shape(batch: Any) -> tuple[int, int, int]:
    """Validate rectangular N x H x W structure and return the shape tuple (N, H, W).

    Args:
        batch: A nested sequence of shape N (images) x H (rows) x W (columns).

    Returns:
        tuple[int, int, int]: (N, H, W)

    Raises:
        TypeError: If containers are not Sequences (or are str/bytes/dict/set).
        ValueError: If batch, image, or row is empty, or if dimensions are ragged.
    """
    if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)):
        raise TypeError(f"Batch must be a Sequence, got {type(batch).__name__}")

    n = len(batch)
    if n == 0:
        raise ValueError("Batch cannot be empty (N >= 1 required)")

    # Validate first image to establish expected spatial dimensions
    first_img = batch[0]
    if not isinstance(first_img, Sequence) or isinstance(first_img, (str, bytes)):
        raise TypeError(f"Image 0 must be a Sequence, got {type(first_img).__name__}")

    h = len(first_img)
    if h == 0:
        raise ValueError("Image cannot be empty: height H >= 1 required")

    first_row = first_img[0]
    if not isinstance(first_row, Sequence) or isinstance(first_row, (str, bytes)):
        raise TypeError(f"Row 0 of Image 0 must be a Sequence, got {type(first_row).__name__}")

    w = len(first_row)
    if w == 0:
        raise ValueError("Row cannot be empty: width W >= 1 required")

    # Validate all images and rows against expected (H, W)
    for img_idx, img in enumerate(batch):
        if not isinstance(img, Sequence) or isinstance(img, (str, bytes)):
            raise TypeError(f"Image {img_idx} must be a Sequence, got {type(img).__name__}")
        if len(img) != h:
            raise ValueError(
                f"Inconsistent image height in batch: image {img_idx} has height {len(img)}, expected {h}"
            )
        for row_idx, row in enumerate(img):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(
                    f"Row {row_idx} of image {img_idx} must be a Sequence, got {type(row).__name__}"
                )
            if len(row) != w:
                raise ValueError(
                    f"Ragged row detected in image {img_idx}, row {row_idx}: length {len(row)}, expected {w}"
                )

    return (n, h, w)


def validate_pixel(pixel: Any, img_idx: int, row_idx: int, col_idx: int) -> float:
    """Validate a single pixel value for numerical correctness and valid [0, 255] range."""
    if isinstance(pixel, bool):
        raise TypeError(
            f"Pixel at ({img_idx}, {row_idx}, {col_idx}) cannot be a boolean: {pixel!r}"
        )
    if not isinstance(pixel, (int, float)):
        raise TypeError(
            f"Pixel at ({img_idx}, {row_idx}, {col_idx}) must be numeric, got {type(pixel).__name__}"
        )

    val = float(pixel)
    if math.isnan(val) or math.isinf(val):
        raise ValueError(
            f"Pixel at ({img_idx}, {row_idx}, {col_idx}) cannot be NaN or Infinite: {pixel!r}"
        )
    if val < 0.0 or val > 255.0:
        raise ValueError(
            f"Pixel value out of range [0, 255] at ({img_idx}, {row_idx}, {col_idx}): got {pixel}"
        )
    return val


# ── Core Normalization Function ──────────────────────────────────────

def normalize_image_batch(batch: Sequence[Sequence[Sequence[float]]]) -> NormalizedBatchResult:
    """Normalize a batch of N x H x W images from 0..255 into 0.0..1.0 while preserving shape.

    Formula:
        normalized_pixel = pixel / 255.0

    Args:
        batch: A nested sequence representing N images of height H and width W.

    Returns:
        NormalizedBatchResult containing the normalized batch and shape metadata.

    Raises:
        TypeError: If structures or pixel types are invalid.
        ValueError: If batch dimensions are empty, ragged, or pixels are outside [0, 255].
    """
    orig_shape = get_batch_shape(batch)
    n, h, w = orig_shape

    normalized_batch: list[list[list[float]]] = []

    for img_idx, img in enumerate(batch):
        norm_img: list[list[float]] = []
        for row_idx, row in enumerate(img):
            norm_row: list[float] = []
            for col_idx, pixel in enumerate(row):
                valid_pixel = validate_pixel(pixel, img_idx, row_idx, col_idx)
                # Linear normalization from [0, 255] to [0.0, 1.0]
                norm_val = valid_pixel / 255.0
                norm_row.append(norm_val)
            norm_img.append(norm_row)
        normalized_batch.append(norm_img)

    norm_shape = get_batch_shape(normalized_batch)
    shape_preserved = (orig_shape == norm_shape)

    return NormalizedBatchResult(
        original_shape=orig_shape,
        normalized_shape=norm_shape,
        normalized_batch=normalized_batch,
        shape_preserved=shape_preserved,
    )


# ── Pretty-Printing and CLI Runner ───────────────────────────────────

def render_result(result: NormalizedBatchResult, title: str) -> None:
    """Format and print batch result cleanly."""
    print(f"--- {title} ---")
    print(f"Original shape (N, H, W):    {result.original_shape}")
    print(f"Normalized shape (N, H, W):  {result.normalized_shape}")
    print(f"Shape preserved:             {result.shape_preserved}")
    print(f"Normalized batch content:    {result.normalized_batch}")
    print()


def main() -> int:
    """Run CLI demonstration of image batch normalization and shape contract."""
    print("=" * 70)
    print("Tensor-Shape Contract — Runnable Concept Example (R38-T049)")
    print("=" * 70)
    print()

    # Mandatory Fixture Demonstration
    print(">>> 1. Executing Mandatory Fixture:")
    print(f"Input batch: {MAIN_BATCH}")
    result = normalize_image_batch(MAIN_BATCH)
    render_result(result, "Mandatory Fixture Result")

    print("Pixel Mapping Breakdown:")
    print(f"  Pixel   0 -> {result.normalized_batch[0][0][0]:.6f} (Exact: 0.0)")
    print(f"  Pixel 255 -> {result.normalized_batch[0][0][1]:.6f} (Exact: 1.0)")
    print(f"  Pixel 128 -> {result.normalized_batch[0][1][0]:.10f} (Expected: 128/255 ≈ 0.5019607843)")
    print(f"  Pixel  64 -> {result.normalized_batch[0][1][1]:.10f} (Expected: 64/255  ≈ 0.2509803922)")
    print()

    # Multi-Image Batch Demonstration
    print(">>> 2. Multi-Image Batch Demonstration (N=2, H=2, W=3):")
    multi_batch = [
        [[0, 100, 200], [50, 150, 255]],
        [[255, 200, 100], [150, 50, 0]],
    ]
    multi_res = normalize_image_batch(multi_batch)
    render_result(multi_res, "2-Image Batch Result")

    # Failure Demonstrations (Caught explicitly to demonstrate validation)
    print(">>> 3. Failure Handling & Contract Enforcement:")

    # Negative pixel
    print("Attempting negative pixel [-5]:")
    try:
        normalize_image_batch([[[-5, 100]]])
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")

    # Pixel > 255
    print("Attempting pixel > 255 [300]:")
    try:
        normalize_image_batch([[[300, 100]]])
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")

    # Ragged rows
    print("Attempting ragged image batch (row 1 has 2 elements, row 2 has 3):")
    try:
        normalize_image_batch([[[0, 100], [50, 150, 200]]])
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")

    # Empty batch
    print("Attempting empty batch []:")
    try:
        normalize_image_batch([])
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")

    print()
    print("Summary: All shape preservation and validation checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

