# Applied Lab R38-T050: Bounding-Box Intersection over Union (IoU)

**Curriculum Module:** `lessons/4-ComputerVision/11-ObjectDetection/`  
**Task ID:** R38-T050  
**Student:** Vaikunth Ankola  

---

## 1. Overview and Objective

In object detection, models predict spatial bounding boxes enclosing targets of interest. Evaluating model quality requires measuring how closely a predicted bounding box aligns with the ground-truth annotation. The primary metric for this geometric overlap is **Intersection over Union (IoU)**, also known as the Jaccard Index.

This applied lab implements an exact, dependency-free reference implementation of 2D axis-aligned bounding-box IoU in Python 3. It explores the standard continuous coordinate convention, verifies core benchmark fixtures, handles boundary geometries (edge touching, corner touching, zero area), and strictly rejects invalid or inverted inputs.

---

## 2. Geometric Theory and Coordinate Convention

### Coordinate Format: Continuous `[x_min, y_min, x_max, y_max]`

Each bounding box is defined by four coordinates:
- `x_min`: Left horizontal boundary
- `y_min`: Top vertical boundary
- `x_max`: Right horizontal boundary
- `y_max`: Bottom vertical boundary

In continuous space:
$$\text{width} = x_{\max} - x_{\min}$$
$$\text{height} = y_{\max} - y_{\min}$$
$$\text{Area} = \text{width} \times \text{height}$$

> **Note on Continuous vs. Discrete Coordinates:**  
> Discrete pixel grids (e.g. legacy PASCAL VOC evaluators) historically added $+1$ to width and height (`x_max - x_min + 1`) under the assumption that pixel indices refer to unit squares. In modern computer vision (e.g. COCO, PyTorch `torchvision`, TensorFlow Object Detection API), coordinates are treated as continuous real numbers, so no $+1$ offset is added.

### Intersection and Union Formulas

Given two boxes $A = [x_1^A, y_1^A, x_2^A, y_2^A]$ and $B = [x_1^B, y_1^B, x_2^B, y_2^B]$:

1. **Intersection Boundaries:**
   $$x_{\text{inter}}^{\min} = \max(x_1^A, x_1^B), \quad y_{\text{inter}}^{\min} = \max(y_1^A, y_1^B)$$
   $$x_{\text{inter}}^{\max} = \min(x_2^A, x_2^B), \quad y_{\text{inter}}^{\max} = \min(y_2^A, y_2^B)$$

2. **Intersection Dimensions:**
   $$\text{inter\_w} = \max(0.0, \, x_{\text{inter}}^{\max} - x_{\text{inter}}^{\min})$$
   $$\text{inter\_h} = \max(0.0, \, y_{\text{inter}}^{\max} - y_{\text{inter}}^{\min})$$
   $$\text{Intersection} = \text{inter\_w} \times \text{inter\_h}$$

3. **Union:**
   $$\text{Union} = \text{Area}(A) + \text{Area}(B) - \text{Intersection}$$

4. **IoU Metric:**
   $$\text{IoU} = \begin{cases} \frac{\text{Intersection}}{\text{Union}}, & \text{if } \text{Union} > 0 \\ 0.0, & \text{if } \text{Union} = 0 \end{cases}$$

---

## 3. Core Fixtures and Scenarios

| Scenario | Box A | Box B | Area A | Area B | Intersection | Union | Expected IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Partial Overlap** | `[0, 0, 2, 2]` | `[1, 1, 3, 3]` | 4.0 | 4.0 | 1.0 | 7.0 | **$1/7 \approx 0.142857$** |
| **Identical Boxes** | `[0, 0, 2, 2]` | `[0, 0, 2, 2]` | 4.0 | 4.0 | 4.0 | 4.0 | **$1.0$** |
| **Disjoint Boxes** | `[0, 0, 1, 1]` | `[2, 2, 3, 3]` | 1.0 | 1.0 | 0.0 | 2.0 | **$0.0$** |
| **Edge-Touching** | `[0, 0, 1, 1]` | `[1, 0, 2, 1]` | 1.0 | 1.0 | 0.0 | 2.0 | **$0.0$** |
| **Corner-Touching** | `[0, 0, 1, 1]` | `[1, 1, 2, 2]` | 1.0 | 1.0 | 0.0 | 2.0 | **$0.0$** |
| **Contained Box** | `[0, 0, 4, 4]` | `[1, 1, 3, 3]` | 16.0 | 4.0 | 4.0 | 16.0 | **$0.25$** |
| **Negative Coords** | `[-2, -2, 0, 0]` | `[-1, -1, 1, 1]`| 4.0 | 4.0 | 1.0 | 7.0 | **$1/7 \approx 0.142857$** |
| **Zero-Area Point** | `[1, 1, 1, 1]` | `[1, 1, 1, 1]` | 0.0 | 0.0 | 0.0 | 0.0 | **$0.0$** |

---

## 4. Boundary Decisions and Validation Policies

1. **Edge or Corner Contact:**  
   Boxes that share an edge or corner touch with zero area ($\text{inter\_w} = 0$ or $\text{inter\_h} = 0$). The intersection area is exactly $0.0$, yielding $\text{IoU} = 0.0$.
2. **Zero-Area Union Sentinel:**  
   When two degenerate boxes (e.g. points or 1D lines) produce $\text{Union} = 0.0$, division by zero is avoided by returning the safe sentinel $0.0$.
3. **Inversion Detection:**  
   Any box where $x_{\min} > x_{\max}$ or $y_{\min} > y_{\max}$ is geometrically invalid and raises a `ValueError`.
4. **Type and Value Enforcement:**  
   Booleans (which inherit from `int` in Python) are explicitly rejected with `TypeError`. Non-finite numbers (`NaN`, `inf`) and non-numeric entries are rejected with `ValueError` or `TypeError`.

---

## 5. How to Run the Example and Tests

### Run the Demonstration

```bash
python3 lessons/4-ComputerVision/11-ObjectDetection/applied-labs/r38-t050/example.py
```

### Run the Automated Unit Test Suite

```bash
python3 -m unittest discover -v -s lessons/4-ComputerVision/11-ObjectDetection/applied-labs/r38-t050 -p "test_*.py"
```

---

## 6. Scope and Educational Limitations

- **Axis-Aligned Only:** The implementation calculates overlap for axis-aligned bounding boxes (AABBs). Rotated bounding boxes or polygon masks require geometric clipping algorithms (e.g. Sutherland-Hodgman) not covered here.
- **Pure Standard Library:** Implemented without external packages (no NumPy, OpenCV, or PyTorch).
- **Offline & Deterministic:** Requires no GPU, network access, or trained model checkpoints.

