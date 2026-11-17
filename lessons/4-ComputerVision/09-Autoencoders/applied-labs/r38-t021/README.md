# Applied Lab R38-T021: Tensor-Shape Contract & Normalization for Autoencoders

## Overview & Lesson Concept Connection

In **Autoencoders** ([Lesson 09: Autoencoders](../../README.md)), neural networks are trained to reconstruct input images through an encoder-decoder bottleneck architecture.

Before feeding image batches (such as MNIST or custom image data) into an autoencoder:
1. **Tensor Shape Invariance:** The batch must adhere to a strict 3D tensor-shape contract of $(N \times H \times W)$ representing $N$ samples, height $H$, and width $W$. Autoencoder network architectures (linear layers or convolutions) require consistent batch and spatial dimensions across the entire input-output pipeline.
2. **Value Domain Normalization:** Raw grayscale images are typically represented as 8-bit integers with pixel values in $[0, 255]$. Neural networks optimize best with normalized floating-point activations in $[0.0, 1.0]$. This aligns directly with reconstruction loss functions (like MSE or Binary Cross Entropy) and decoder activation functions (such as `Sigmoid`).

This lab provides a runnable concept implementation and test suite demonstrating how input tensors are validated against structural contracts and normalized without altering their shape.

---

## Specifications

- **Input:** $N \times H \times W$ numeric pixels in $[0, 255]$.
- **Output:** Exact $N \times H \times W$ shape with float values in $[0.0, 1.0]$.
  - Example: Batch `[[[0, 255], [128, 64]]]` (shape $(1, 2, 2)$) normalizes to `[[[0.0, 1.0], [0.5019..., 0.2509...]]]` retaining shape $(1, 2, 2)$.
- **Validation Rules:**
  - Rejects pixel values $< 0$ or $> 255$.
  - Rejects non-numeric types (e.g., strings, booleans, `None`, NaNs).
  - Rejects ragged or inconsistent dimensions.
  - Rejects invalid ranks (must be 3D).

---

## Files

- `example.py`: Core implementation of `validate_and_normalize_tensor` and `get_shape`, along with a runnable demonstration CLI.
- `test_example.py`: Comprehensive test suite verifying shape preservation, value bounds, type checking, and edge case rejections.
- `README.md`: Lab documentation and conceptual explanation.

---

## Running the Code

### Execute Runnable Concept Demonstration
```bash
python3 lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t021/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t021 -p "test_*.py"
```
