"""
Tensor-Shape Contract — Offline Service Adapter
Applied lab for lessons/4-ComputerVision/10-GANs (Task R38-T015)

Concept & Contract:
In production computer vision pipelines, image preprocessing services may be accessed
via cloud APIs or offline services. To ensure reliable local development and unit testing,
the external dependency is placed behind a callable interface (`ImagePreprocessorService`),
and a deterministic local adapter (`LocalPreprocessorAdapter`) is supplied.

Contract:
- Accept N x H x W numeric pixels in range [0, 255].
- Normalize to [0.0, 1.0] without changing batch shape.
- Batch [[[0, 255], [128, 64]]] stays shape (1, 2, 2); 0 -> 0.0, 255 -> 1.0.
- Negative (< 0) or > 255 pixels are rejected (raises ValueError).

Focus:
- Put the external dependency behind a callable interface.
- Supply a deterministic local offline adapter.
- Identify which real-service behaviors this mock does not establish (e.g. network latency,
  authentication, remote rate limits, hardware memory saturation, distributed serialization).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

Number = Union[int, float]
Image2D = List[List[Number]]
Batch3D = List[Image2D]
Shape = Tuple[int, int, int]


def get_shape(batch: Batch3D) -> Shape:
    """Validate rectangular structure and return (N, H, W)."""
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
            raise ValueError("All images in batch must have identical height H.")
        for row in img:
            if not isinstance(row, list) or len(row) != w:
                raise ValueError("All rows in image must have identical width W.")
    return (n, h, w)


class ImagePreprocessorService(ABC):
    """Callable interface abstracting preprocessing operations."""

    @abstractmethod
    def preprocess_batch(self, batch: Batch3D) -> Batch3D:
        """Validate and normalize batch from [0, 255] to [0.0, 1.0]."""
        pass


class LocalPreprocessorAdapter(ImagePreprocessorService):
    """
    Deterministic offline adapter fulfilling ImagePreprocessorService contract.

    Limitations:
    This adapter runs locally and deterministically. It does NOT establish:
    1. Remote network latency, retries, or transient connection dropouts.
    2. Cloud authentication, token expiry, or API rate limiting.
    3. GPU device allocation or CUDA out-of-memory behavior.
    """

    def preprocess_batch(self, batch: Batch3D) -> Batch3D:
        shape = get_shape(batch)

        normalized_batch: Batch3D = []
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
            normalized_batch.append(norm_img)

        out_shape = get_shape(normalized_batch)
        if shape != out_shape:
            raise RuntimeError("Shape mismatch between input and normalized output.")

        return normalized_batch


def run_demonstration():
    contract_batch: Batch3D = [[[0, 255], [128, 64]]]
    adapter: ImagePreprocessorService = LocalPreprocessorAdapter()
    normalized = adapter.preprocess_batch(contract_batch)

    print("=== Tensor-Shape Contract: Offline Service Adapter Demo (R38-T015) ===")
    print(f"Input: {contract_batch} (Shape: {get_shape(contract_batch)})")
    print(f"Normalized: {normalized} (Shape: {get_shape(normalized)})")
    print("Precondition & Invariant: 0 -> 0.0, 255 -> 1.0, shape strictly retained.")


if __name__ == "__main__":
    run_demonstration()
