ROC vs PR curve is one of those topics where interviewers aren’t testing definitions — they’re testing whether you understand **probability ranking, class imbalance, and decision thresholds in production systems**.

I’ll explain it like you would design a real ML system.

---

# 1. What problem are ROC and PR curves solving?

Both curves evaluate:

> “How good is my model at ranking positive examples higher than negative ones across all thresholds?”

Instead of picking one threshold (like 0.5), we sweep all thresholds.

---

# 2. ROC Curve (Receiver Operating Characteristic)

### What it plots:

* **TPR (Recall / Sensitivity)** vs **FPR**

### Formulas:

* TPR = TP / (TP + FN)
* FPR = FP / (FP + TN)

### Intuition (senior-level):

ROC asks:

> “Out of all actual positives, how many did I catch, and how many false alarms did I generate?”

So ROC is about **ranking quality independent of class imbalance**.

---

## ROC Curve Behavior

* Ideal model: TPR → 1, FPR → 0
* Random model: diagonal line

---

# 3. PR Curve (Precision-Recall Curve)

### What it plots:

* **Precision vs Recall**

### Formulas:

* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)

### Intuition:

PR asks:

> “When my model says positive, how often is it actually correct?”

So PR focuses on **positive class quality**, not overall ranking.

---

# 4. ROC vs PR — the real engineering difference

## When ROC is misleading

If dataset is imbalanced:

Example:

* 99,000 negatives
* 1,000 positives

A dumb model predicting everything negative:

* FPR still looks very small
* ROC looks deceptively good

👉 Because TN dominates denominator

---

## PR curve is more honest here

Because:

* It ignores TN completely
* Focuses only on positive prediction quality

---

## Senior rule of thumb:

| Scenario                         | Use |
| -------------------------------- | --- |
| Balanced dataset                 | ROC |
| Highly imbalanced dataset        | PR  |
| Fraud detection                  | PR  |
| Medical diagnosis                | PR  |
| Ranking / general classification | ROC |

---

# 5. Code Example (ROC + PR in Python)

We’ll use `sklearn`.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
```

---

## Step 1: Create imbalanced dataset

```python
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    weights=[0.98, 0.02],   # heavy imbalance
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

---

## Step 2: Train model

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_scores = model.predict_proba(X_test)[:, 1]
```

---

# 6. ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

# 7. Precision-Recall Curve

```python
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
```

---

# 8. What you should say in interviews (important)

A strong answer sounds like:

> ROC measures ranking quality using TPR vs FPR, but can be misleading under class imbalance because TN dominates FPR.
>
> PR curve directly evaluates performance on the positive class and is more sensitive to false positives in imbalanced datasets.

---

# 9. Deep intuition (what senior engineers care about)

## ROC = “global ranking quality”

It asks:

> “Did positives rank above negatives overall?”

## PR = “business utility of positive predictions”

It asks:

> “Are my positive predictions actually useful?”

---

# 10. Bonus: AUC interpretation

* ROC-AUC = probability a random positive is ranked higher than a random negative
* PR-AUC = quality of positive retrieval

---

# If you want next level (senior interview depth)

I can also explain:

* Why PR curve is non-monotonic sometimes
* How threshold tuning is done in production (cost-based optimization)
* Why AUC can be misleading even in ROC
* How calibration changes ROC/PR behavior
* Multi-class ROC/PR strategies

Just tell me.
