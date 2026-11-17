# Tensor-Shape Contract — Runnable Concept Example (R38-T049)

## 1. Task Objective

In Computer Vision, images are represented numerically as multidimensional tensors. Before passing images to neural network architectures (such as Convolutional Neural Networks), images are organized into batches and preprocessed.

A critical preprocessing requirement is the **Tensor-Shape Contract**:
1. **Pixel Normalization:** Transforming raw 8-bit color intensities from integer range $[0, 255]$ into floating-point range $[0.0, 1.0]$.
2. **Shape Preservation:** Ensuring that the batch dimensions ($N \times H \times W$) are strictly preserved during transformation to avoid shape mismatch errors in neural network layers.

This applied software-engineering lab demonstrates a minimal, complete, and robust implementation of batch shape validation and pixel normalization using only Python's standard library.

---

## 2. Core Computer Vision Concepts

### What is $N \times H \times W$?
In digital image processing, a grayscale image is a 2D matrix of pixels of size:
$$\text{Height } (H) \times \text{Width } (W)$$
When multiple images are processed concurrently (minibatch processing), they are bundled into a 3D tensor of shape:
$$N \times H \times W$$
Where:
- $N$: Batch size (number of independent images).
- $H$: Spatial height (number of rows).
- $W$: Spatial width (number of columns / pixels per row).

### Why Pixel Normalization is Used
Raw image sensors encode intensity using 8 bits per channel ($2^8 = 256$ discrete levels), resulting in pixel values between $0$ (black) and $255$ (white).
Feeding raw large integer values ($0\text{--}255$) into deep neural networks causes gradient explosion and slows down optimization. Normalizing pixels to $[0.0, 1.0]$ standardizes scale and stabilizes gradient descent training.

### Normalization Formula
$$\text{normalized\_pixel} = \frac{\text{pixel}}{255.0}$$

---

## 3. Mandatory Specification Fixture

### Input Batch
```python
batch = [
    [
        [0, 255],
        [128, 64]
    ]
]
```

### Shape and Normalization Mapping
- **Original Shape:** `(1, 2, 2)` ($N=1, H=2, W=2$)
- **Normalized Shape:** `(1, 2, 2)` (strictly preserved)
- **Pixel Values:**
  - $0 \to \frac{0}{255} = 0.0$
  - $255 \to \frac{255}{255} = 1.0$
  - $128 \to \frac{128}{255} \approx 0.5019607843$
  - $64 \to \frac{64}{255} \approx 0.2509803922$

### Output Batch
```python
[
    [
        [0.0, 1.0],
        [0.5019607843137255, 0.25098039215686274]
    ]
]
```

---

## 4. Contract and Validation Rules

### Pixel Validation
- **Valid Range:** $0.0 \le \text{pixel} \le 255.0$ (supports both integers and floats).
- **Rejected Inputs:**
  - Negative values ($< 0$): raises `ValueError`.
  - Values greater than 255 ($> 255$): raises `ValueError`.
  - Non-finite numbers (`NaN`, `+inf`, `-inf`): raises `ValueError`.
  - Booleans (`True`/`False`): raises `TypeError` (because Python `bool` is a subclass of `int`, but is not a valid pixel intensity).
  - Non-numeric types (e.g. strings, objects): raises `TypeError`.

### Shape & Hierarchy Validation
- **Rectangularity:** All images in a batch must have identical height $H$, and all rows across all images must have identical width $W$.
- **Rejected Structures:**
  - Empty batch `[]`: raises `ValueError`.
  - Empty image `[[]]`: raises `ValueError`.
  - Empty row `[[[]]]`: raises `ValueError`.
  - Ragged rows (e.g. row 0 has 2 pixels, row 1 has 3 pixels): raises `ValueError`.
  - Non-sequence outer or inner containers: raises `TypeError`.
- **Immutability:** The input nested list structure is never mutated in-place; a new normalized structure is returned.

---

## 5. Scope and Educational Limitations

This lab is a local preprocessing demonstration. It does **not**:
1. Load image files from disk (JPEG, PNG) or call OpenCV, Pillow, or imageio.
2. Call numerical tensor libraries like NumPy, PyTorch, or TensorFlow.
3. Train or evaluate a computer vision model.
4. Require GPUs, cloud vision services, or camera hardware.

---

## 6. Execution Instructions

### Prerequisites
- Python 3.8+ (standard library only; no external dependencies required).

### Run the Example
```bash
python3 lessons/4-ComputerVision/06-IntroCV/applied-labs/r38-t049/example.py
```

### Run Automated Tests
```bash
python3 -m unittest discover -v -s lessons/4-ComputerVision/06-IntroCV/applied-labs/r38-t049 -p "test_*.py"
```

### Expected Output
```
======================================================================
Tensor-Shape Contract — Runnable Concept Example (R38-T049)
======================================================================

>>> 1. Executing Mandatory Fixture:
Input batch: [[[0, 255], [128, 64]]]
--- Mandatory Fixture Result ---
Original shape (N, H, W):    (1, 2, 2)
Normalized shape (N, H, W):  (1, 2, 2)
Shape preserved:             True
Normalized batch content:    [[[0.0, 1.0], [0.5019607843137255, 0.25098039215686274]]]

Pixel Mapping Breakdown:
  Pixel   0 -> 0.000000 (Exact: 0.0)
  Pixel 255 -> 1.000000 (Exact: 1.0)
  Pixel 128 -> 0.5019607843 (Expected: 128/255 ≈ 0.5019607843)
  Pixel  64 -> 0.2509803922 (Expected: 64/255  ≈ 0.2509803922)
...
```
