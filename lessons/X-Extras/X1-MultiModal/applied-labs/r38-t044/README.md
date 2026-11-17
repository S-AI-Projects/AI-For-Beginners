# Applied Lab R38-T044: Binary Classification Evaluation — Worked Example

## 1. Task Objective

Compute a confusion matrix, precision and recall for a binary classifier using
fixed predicted and true labels. Make the zero-denominator policy explicit.

> **Simulation Disclaimer:** This is a local educational example using fixed
> label fixtures from the Python standard library. It does not train a model,
> perform real inference, evaluate a production classifier, or use any paid
> API, GPU or cloud service.

---

## 2. What Is Binary Classification?

A **binary classifier** assigns each sample to one of two classes:

| Label | Meaning |
|:-----:|:--------|
| **1** | Positive (e.g. "spam", "cat detected", "disease present") |
| **0** | Negative (e.g. "not spam", "no cat", "healthy") |

After running a classifier, every sample has two labels:

- **Actual label (`y_true`):** The ground-truth answer.
- **Predicted label (`y_pred`):** What the classifier guessed.

---

## 3. Confusion Matrix

Comparing each actual/predicted pair produces four categories:

|  | **Predicted = 1** | **Predicted = 0** |
|---|:---:|:---:|
| **Actual = 1** | True Positive (**TP**) | False Negative (**FN**) |
| **Actual = 0** | False Positive (**FP**) | True Negative (**TN**) |

- **TP:** Correctly identified a positive.
- **FN:** Missed a positive (predicted negative when actually positive).
- **FP:** False alarm (predicted positive when actually negative).
- **TN:** Correctly identified a negative.

**Invariant:** TP + FN + FP + TN = total number of samples.

---

## 4. Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP}$$

> *"Of everything the classifier called positive, how many were actually positive?"*

$$\text{Recall} = \frac{TP}{TP + FN}$$

> *"Of all actual positives, how many did the classifier find?"*

---

## 5. Worked Example — Main Fixture

### Input

```
y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]
```

### Step-by-Step Classification

| Index | Actual (`y_true`) | Predicted (`y_pred`) | Classification |
|:-----:|:-----------------:|:--------------------:|:--------------:|
| 0     | 1                 | 1                    | **TP**         |
| 1     | 1                 | 0                    | **FN**         |
| 2     | 0                 | 1                    | **FP**         |
| 3     | 0                 | 0                    | **TN**         |

### Confusion Matrix

```
TP = 1    FN = 1
FP = 1    TN = 1
```

### Metric Calculation

```
Precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
Recall    = TP / (TP + FN) = 1 / (1 + 1) = 0.5
```

### Expected Output

```
TP = 1
FN = 1
FP = 1
TN = 1
Precision = 0.5
Recall = 0.5
Total samples = 4
```

---

## 6. Zero-Denominator Problem and Policy

### The Problem

When the classifier predicts **no positives at all** (`y_pred` is all zeros),
TP = 0 and FP = 0, so:

$$\text{Precision} = \frac{0}{0 + 0} = \frac{0}{0} \quad \text{(undefined!)}$$

### The Policy

This example uses a **return-zero policy**:

```python
if tp + fp == 0:
    precision = 0.0
else:
    precision = tp / (tp + fp)
```

The same policy applies to recall when there are no actual positives
(TP + FN = 0).

**Why return 0.0?** Returning 0.0 clearly signals "the metric could not
be meaningfully computed" without crashing the program with a
`ZeroDivisionError` or producing `NaN`/`Inf`, which would propagate
silently through downstream calculations.

### Zero-Denominator Fixture

```
y_true = [1, 1, 0, 0]
y_pred = [0, 0, 0, 0]
```

| Index | Actual | Predicted | Classification |
|:-----:|:------:|:---------:|:--------------:|
| 0     | 1      | 0         | **FN**         |
| 1     | 1      | 0         | **FN**         |
| 2     | 0      | 0         | **TN**         |
| 3     | 0      | 0         | **TN**         |

```
TP = 0    FN = 2
FP = 0    TN = 2
Precision = 0.0   (zero-denominator policy)
Recall = 0.0
```

---

## 7. Prerequisites

- **Python 3.8+** (standard library only).
- No external packages, model weights, GPU, or cloud service required.

---

## 8. Execution Commands

### Run the Example

```bash
python3 lessons/X-Extras/X1-MultiModal/applied-labs/r38-t044/example.py
```

### Expected Terminal Output

```
============================================================
Fixture: Main
y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]
------------------------------------------------------------
TP = 1
FN = 1
FP = 1
TN = 1
Precision = 0.5
Recall = 0.5
Total samples = 4
Confusion-matrix sum check: 4 == 4
============================================================

============================================================
Fixture: No positive predictions (zero-denominator)
y_true = [1, 1, 0, 0]
y_pred = [0, 0, 0, 0]
------------------------------------------------------------
TP = 0
FN = 2
FP = 0
TN = 2
Precision = 0.0
Recall = 0.0
Total samples = 4
Confusion-matrix sum check: 4 == 4
============================================================
```

### Run Automated Tests

```bash
python3 -m unittest discover -s lessons/X-Extras/X1-MultiModal/applied-labs/r38-t044 -p "test_*.py"
```

Verbose:
```bash
python3 -m unittest discover -s lessons/X-Extras/X1-MultiModal/applied-labs/r38-t044 -p "test_*.py" -v
```

---

## 9. Limitations

1. Uses **fixed label lists**, not predictions from a trained model.
2. Only binary classification (two classes: 0 and 1).
3. Does not compute F1-score, accuracy, AUC-ROC, or multi-class metrics.
4. Does not perform cross-validation or statistical significance testing.
5. The zero-denominator policy (return 0.0) is one of several valid
   conventions; scikit-learn, for instance, can return 0.0 with a warning.

