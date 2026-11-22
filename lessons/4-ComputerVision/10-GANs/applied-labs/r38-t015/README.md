# Applied Lab R38-T015: Tensor-Shape Contract — Offline Service Adapter

## Overview

In production CV applications (see [GANs](../../README.md)), image preprocessing components are decoupled using interface adapters. This lab places preprocessing behind an abstract `ImagePreprocessorService` interface and supplies a deterministic `LocalPreprocessorAdapter`.

## Contract & Specification

- **Input Format:** $N \times H \times W$ numeric pixels in $[0, 255]$.
- **Contract Fixture:** `[[[0, 255], [128, 64]]]` (shape: `(1, 2, 2)`).
- **Normalized Output:** Preserves shape `(1, 2, 2)`; maps `0` $\to$ `0.0`, `255` $\to$ `1.0`.
- **Validation:** Values $< 0$ or $> 255$ raise `ValueError`.

## Service Adapter & Scope Limitations

> [!NOTE]
> **What This Offline Mock Does NOT Establish:**
> 1. Remote service latency, network jitter, or socket failures.
> 2. Cloud IAM authentication or token refresh lifecycles.
> 3. GPU device memory limits or distributed sharding.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/10-GANs/applied-labs/r38-t015 -p "test_*.py" -v
```
