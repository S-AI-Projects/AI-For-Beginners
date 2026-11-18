# Applied Lab R38-T060: Training-Only Missing Value Imputation

## Overview
Demonstrates proper fit/transform separation for missing-value imputation in machine learning preprocessing, ensuring validation data is never used to estimate imputation statistics.

## Mathematical Formulation
- Training Observations: $X_{train} = [2, 	ext{None}, 4]$
- Fitted Mean: $\mu = rac{2 + 4}{2} = 3.0$
- Validation Input: $X_{val} = [100, 	ext{None}]$
- Imputed Validation: $[100, 3.0]$ (not $[100, 100]$, preventing data leakage).
