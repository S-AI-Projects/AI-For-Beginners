"""
Tensor-shape contract & normalization for Autoencoder input preprocessing.

In autoencoders (such as those for MNIST or image reconstruction/denoising),
the encoder network expects normalized batches of numeric images with consistent
spatial dimensions (N x H x W). Each pixel value in [0, 255] is mapped to [0.0, 1.0]
to ensure stable gradient descent, matching activation ranges (e.g., Sigmoid outputs),
while strictly preserving the batch and spatial dimensions (shape invariance).
"""

from typing import Any, List, Sequence, Tuple, Union


def get_shape(data: Any) -> Tuple[int, ...]:
    """
    Recursively determine the shape of nested sequences.
    Raises ValueError if sub-sequences have inconsistent lengths or non-sequences are nested improperly.
    """
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        first_elem = data[0]
        first_subshape = get_shape(first_elem) if isinstance(first_elem, (list, tuple)) else ()
        for elem in data[1:]:
            elem_subshape = get_shape(elem) if isinstance(elem, (list, tuple)) else ()
            if elem_subshape != first_subshape:
                raise ValueError(
                    f"Inconsistent dimensions detected: expected sub-shape {first_subshape}, got {elem_subshape}"
                )
        return (len(data),) + first_subshape
    return ()


def validate_and_normalize_tensor(
    tensor: Sequence[Sequence[Sequence[Union[int, float]]]]
) -> List[List[List[float]]]:
    """
    Validates and normalizes an input tensor of shape (N, H, W).

    Lesson Concept Connection:
    In autoencoders (lessons/4-ComputerVision/09-Autoencoders), raw input images typically
    arrive as 8-bit integer grayscale batches of dimensions N x H x W with pixel values
    in [0, 255]. Autoencoders trained with reconstruction loss (such as MSE or BCE) require
    inputs normalized to [0.0, 1.0]. This preprocessing contract ensures:
      1. Structural integrity: Exactly 3 dimensions (Batch size N, Height H, Width W) with N, H, W > 0.
      2. Domain validity: All pixel values are numeric and within [0, 255].
      3. Normalization: Values are mapped to [0.0, 1.0] via `pixel / 255.0` without altering tensor shape.

    Args:
        tensor: Nested sequence representing an N x H x W image batch.

    Returns:
        A 3D nested list of shape (N, H, W) containing float values in [0.0, 1.0].

    Raises:
        TypeError: If tensor or any element is not a sequence/number, or contains boolean/invalid types.
        ValueError: If dimensions are not 3D (N x H x W), dimensions are empty/ragged,
                    or pixel values fall outside [0, 255].
    """
    # Reject numpy arrays if numpy is used, or convert them if appropriate, but standard sequences:
    # First check outer type
    if not isinstance(tensor, (list, tuple)):
        # Check if numpy array
        try:
            import numpy as np
            if isinstance(tensor, np.ndarray):
                tensor = tensor.tolist()
            else:
                raise TypeError(f"Expected sequence representing 3D tensor, got {type(tensor).__name__}")
        except ImportError:
            raise TypeError(f"Expected sequence representing 3D tensor, got {type(tensor).__name__}")

    # Validate shape consistency and depth
    shape = get_shape(tensor)
    if len(shape) != 3:
        raise ValueError(f"Expected 3D tensor of shape (N, H, W), got shape with {len(shape)} dimension(s): {shape}")

    n, h, w = shape
    if n == 0 or h == 0 or w == 0:
        raise ValueError(f"Dimensions must be positive non-zero, got shape: {shape}")

    # Validate pixel values and normalize
    normalized_batch: List[List[List[float]]] = []
    for i, image in enumerate(tensor):
        normalized_image: List[List[float]] = []
        for r, row in enumerate(image):
            normalized_row: List[float] = []
            for c, val in enumerate(row):
                # Note: bool is a subclass of int in Python, so isinstance(True, int) is True!
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(
                        f"Non-numeric pixel value at [{i}][{r}][{c}]: {val!r} of type {type(val).__name__}"
                    )
                # Check for NaN / Inf
                if isinstance(val, float) and (val != val or val in (float('inf'), float('-inf'))):
                    raise ValueError(f"Invalid non-finite pixel value at [{i}][{r}][{c}]: {val}")

                if val < 0 or val > 255:
                    raise ValueError(
                        f"Pixel value at [{i}][{r}][{c}] out of range [0, 255]: {val}"
                    )

                # Normalize 0 -> 0.0, 255 -> 1.0
                norm_val = float(val) / 255.0
                normalized_row.append(norm_val)
            normalized_image.append(normalized_row)
        normalized_batch.append(normalized_image)

    return normalized_batch


def main() -> None:
    """Runnable demonstration showing tensor contract validation & normalization."""
    print("=== Autoencoder Input Preprocessing: Tensor-Shape Contract & Normalization ===")
    sample_batch = [[[0, 255], [128, 64]]]
    print(f"Original input batch: {sample_batch}")
    print(f"Input shape: {get_shape(sample_batch)} (1 sample, height=2, width=2)")

    normalized = validate_and_normalize_tensor(sample_batch)
    print(f"Normalized batch: {normalized}")
    print(f"Normalized shape: {get_shape(normalized)} (Shape strictly preserved)")

    # Test edge values
    assert normalized[0][0][0] == 0.0, "0 must map to 0.0"
    assert normalized[0][0][1] == 1.0, "255 must map to 1.0"
    print("Contract verified successfully: pixel values in [0, 1] without altering shape.")


if __name__ == "__main__":
    main()
