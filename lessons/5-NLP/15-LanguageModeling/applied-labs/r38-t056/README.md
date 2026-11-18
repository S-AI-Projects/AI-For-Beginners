# Applied Lab R38-T056: Refactored Cosine-Similarity Lookup

## Overview
Refactors the vector similarity example in Language Modeling into a pure computational engine and an independent presentation layer.

## Architecture
- `compute_cosine_similarity`: Pure mathematical function with zero side effects.
- `format_similarity_report`: Formatting presentation layer for user-facing output.
- `UNDEFINED_SENTINEL`: Clear sentinel (`None`) returned when comparing against zero vectors.
