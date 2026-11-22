# Applied Lab R38-T016: Segmentation Mask Overlap & Component Refactor

## Overview

In Computer Vision Semantic Segmentation (see [Segmentation](../../README.md)), models classify individual pixels into discrete labels. Accuracy is evaluated by computing Intersection over Union (IoU) per class:

$$\text{IoU}(c) = \frac{|\text{Truth}_c \cap \text{Pred}_c|}{|\text{Truth}_c \cup \text{Pred}_c|}$$

## Contract & Specification

- **Contract Fixture:**
  - `truth = [0, 1, 1, 0]`
  - `prediction = [0, 1, 0, 0]`
- **Class 1 Overlap:** Intersection = 1, Union = 2 $\to \text{IoU} = 1/2 = 0.5$.
- **Class 0 Overlap:** Intersection = 2, Union = 3 $\to \text{IoU} = 2/3 \approx 0.6667$.
- **Absent-Class Policy:** When a class is absent from both ground truth and prediction ($\text{Union} = 0$), IoU is explicitly defined as `None`.

## Refactoring Architecture

1. **Baseline Monolith (`calculate_iou_baseline`):** Couples mask iteration, histogram aggregation, and division into a single loop.
2. **Refactored Architecture (`SegmentationEvaluator` + `BinaryOverlapMetrics`):**
   - Extracts binary metric calculation (`BinaryOverlapMetrics.compute_iou`) as an isolated, reusable single responsibility.
   - Proves strict before/after behavioral equivalence.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t016 -p "test_*.py" -v
```
