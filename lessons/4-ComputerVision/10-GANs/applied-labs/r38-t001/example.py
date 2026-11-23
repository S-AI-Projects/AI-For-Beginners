"""
Applied Lab R38-T001: Tensor-Shape Contract and Image Batch Normalization
Lesson: lessons/4-ComputerVision/10-GANs

Contract:
- Accept N x H x W numeric pixels in 0..255
- Produce normalized values in 0..1 without changing batch shape
- Batch [[[0, 255], [128, 64]]] stays shape (1, 2, 2)
- 0 -> 0.0 and 255 -> 1.0
- Pixels < 0 or > 255 are rejected with ValueError
"""

from typing import List, Tuple, Any, Union

Numeric = Union[int, float]
Image = List[List[Numeric]]
Batch = List[Image]


def get_batch_shape(batch: Any) -> Tuple[int, int, int]:
    """
    Validate and return the 3D shape (N, H, W) of a batch.
    Raises ValueError or TypeError if input is not a regular 3D list.
    """
    if not isinstance(batch, list):
        raise TypeError(f"Batch must be a list, got {type(batch).__name__}")
    if len(batch) == 0:
        raise ValueError("Batch must be non-empty")
    
    n = len(batch)
    if not isinstance(batch[0], list):
        raise ValueError(f"Batch elements must be 2D images (lists), got {type(batch[0]).__name__}")
    if len(batch[0]) == 0:
        raise ValueError("Image must have height H > 0")
    
    h = len(batch[0])
    if not isinstance(batch[0][0], list):
        raise ValueError(f"Image rows must be lists, got {type(batch[0][0]).__name__}")
    if len(batch[0][0]) == 0:
        raise ValueError("Image must have width W > 0")
    
    w = len(batch[0][0])

    for img_idx, img in enumerate(batch):
        if not isinstance(img, list) or len(img) != h:
            raise ValueError(f"Inconsistent height in batch at index {img_idx}: expected {h}")
        for row_idx, row in enumerate(img):
            if not isinstance(row, list) or len(row) != w:
                raise ValueError(
                    f"Inconsistent width at batch {img_idx}, row {row_idx}: expected {w}"
                )
    return (n, h, w)


def validate_and_normalize_batch(batch: Batch) -> List[List[List[float]]]:
    """
    Validates input batch and normalizes pixel values from [0, 255] to [0.0, 1.0].
    
    Parameters:
        batch: 3D nested list of shape (N, H, W) with numeric pixels in range [0, 255]
        
    Returns:
        Normalized 3D nested list of shape (N, H, W) with float values in [0.0, 1.0]
        
    Raises:
        ValueError: If shape is irregular or pixel values are out of [0, 255]
        TypeError: If pixels or batch structure are not valid numeric/list types
    """
    n, h, w = get_batch_shape(batch)
    normalized_batch: List[List[List[float]]] = []

    for img_idx, img in enumerate(batch):
        norm_img: List[List[float]] = []
        for row_idx, row in enumerate(img):
            norm_row: List[float] = []
            for col_idx, val in enumerate(row):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise TypeError(
                        f"Pixel at ({img_idx}, {row_idx}, {col_idx}) must be numeric, got {type(val).__name__}"
                    )
                if val < 0 or val > 255:
                    raise ValueError(
                        f"Pixel value {val} at ({img_idx}, {row_idx}, {col_idx}) out of range [0, 255]"
                    )
                norm_row.append(float(val) / 255.0)
            norm_img.append(norm_row)
        normalized_batch.append(norm_img)

    return normalized_batch


if __name__ == "__main__":
    sample_batch = [[[0, 255], [128, 64]]]
    print("Input shape:", get_batch_shape(sample_batch))
    normalized = validate_and_normalize_batch(sample_batch)
    print("Normalized batch:", normalized)
    print("Output shape:", get_batch_shape(normalized))
