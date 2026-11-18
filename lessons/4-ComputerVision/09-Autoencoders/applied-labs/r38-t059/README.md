# Applied Lab R38-T059: Tensor-Shape Contract and Normalization

## Overview
Demonstrates input validation and channel normalization for autoencoder preprocessing pipelines, preserving (N, H, W) tensor dimensions while mapping pixels into [0.0, 1.0].

## Contract & Preprocessing Rules
- Input: 3D nested array of shape `(N, H, W)` with pixel values in `[0, 255]`.
- Output: Matching shape `(N, H, W)` with pixel values in `[0.0, 1.0]`.
- Boundary Enforcement: Negative values or values $> 255$ are rejected with `ValueError`.
