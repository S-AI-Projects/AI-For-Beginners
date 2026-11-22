# Applied Lab R38-T018: Segmentation Mask Overlap — Failure and Recovery

## Overview

In Computer Vision Segmentation pipelines (see [Segmentation](../../README.md)), evaluation services must compute per-class Intersection-over-Union (IoU) while gracefully catching corrupted inputs (e.g., mask dimension mismatches) and immediately recovering for subsequent valid frames.

## Contract & Specification

- **Input Masks:** Integer sequences representing class IDs per pixel.
- **Contract Fixture:**
  - `truth = [0, 1, 1, 0]`
  - `prediction = [0, 1, 0, 0]`
- **Outcomes:**
  - Class 1 IoU: `1/2` (`0.5`)
  - Class 0 IoU: `2/3` ($\approx 0.6667$)
  - Absent Class Policy: `None` when absent from both masks.

## Failure & Recovery Focus

When an invalid input (e.g. Length 3 vs 4) is supplied:
1. An explicit `ValueError` is raised with a diagnostic message detailing the mismatch.
2. The pipeline maintains no corrupted internal state.
3. The very next valid input processes normally without residual side effects.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t018 -p "test_*.py" -v
```
