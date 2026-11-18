# Applied Lab R38-T058: Single Perceptron Update Step

## Overview
Demonstrates one perceptron update step on a 2D input fixture, adhering to a documented zero-score tie rule, with failure handling for invalid inputs and state recovery.

## Mathematics & Rules
- Activation: $	ext{score} = w_0 x_0 + w_1 x_1 + b$
- Zero-Score Tie Policy: $	ext{score} \le 0 \implies \hat{y} = 0$; $	ext{score} > 0 \implies \hat{y} = 1$.
- Weight Update: $w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$
- Bias Update: $b \leftarrow b + \eta \cdot (y - \hat{y})$
