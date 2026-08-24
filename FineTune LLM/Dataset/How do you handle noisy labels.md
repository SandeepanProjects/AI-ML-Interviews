# 1. How do you handle noisy labels?

## What are noisy labels?

A **noisy label** means the training target is incorrect, inconsistent, ambiguous, or low quality.

Example for a sentiment classification dataset:

```text
Text: "This product is amazing!"

Correct label: POSITIVE

But dataset label: NEGATIVE  ❌
```

For LLM fine-tuning:

```text
User:
What is LoRA?

Expected answer:
LoRA is a parameter-efficient fine-tuning technique.

Dataset answer:
LoRA updates every parameter of the model. ❌
```

If you train on bad labels, the model learns bad behavior.

---

## Types of noisy labels

### 1. Incorrect labels

```text
Input:
"Excellent service"

Label:
NEGATIVE ❌
```

### 2. Inconsistent labels

```text
Example 1:
"Good product" → POSITIVE

Example 2:
"Good product" → NEGATIVE ❌
```

### 3. Ambiguous labels

```text
Text:
"The product is okay."

Label:
POSITIVE?
NEUTRAL?
```

Different annotators may disagree.

### 4. Low-quality LLM responses

For instruction tuning:

```text
Question:
Explain LoRA.

Answer:
LoRA is good.
```

Technically related, but poor training quality.

### 5. Hallucinated synthetic labels

```text
Question:
Who invented X?

Synthetic answer:
Incorrect fabricated information ❌
```

---

# 2. How do you detect noisy labels?

Use multiple methods.

```text
Dataset
   │
   ├── Rule-based validation
   │
   ├── Duplicate/conflict detection
   │
   ├── Model confidence
   │
   ├── Cross-model validation
   │
   ├── LLM-as-a-judge
   │
   └── Human review
```

---

## Method 1: Rule-based validation

For classification:

```python
VALID_LABELS = {
    "positive",
    "negative",
    "neutral"
}


def is_valid_label(example):

    label = example["label"].lower()

    return label in VALID_LABELS
```

Usage:

```python
example = {
    "text": "Amazing product",
    "label": "unknown"
}

print(is_valid_label(example))
```

Output:

```text
False
```

For fine-tuning, validate output structure.

Example:

```python
def is_valid_answer(example):

    answer = example["answer"].strip()

    if len(answer) < 10:
        return False

    return True
```

---

# 3. Method 2: Detect conflicting labels

Suppose:

```python
dataset = [
    {
        "text": "I love this product",
        "label": "positive"
    },
    {
        "text": "I love this product",
        "label": "negative"
    }
]
```

Detect conflicts:

```python
from collections import defaultdict


def find_conflicts(dataset):

    labels_by_text = defaultdict(set)

    for item in dataset:

        text = item["text"].strip().lower()

        label = item["label"]

        labels_by_text[text].add(label)

    conflicts = {}

    for text, labels in labels_by_text.items():

        if len(labels) > 1:

            conflicts[text] = labels

    return conflicts
```

Usage:

```python
conflicts = find_conflicts(dataset)

print(conflicts)
```

Output:

```text
{
    'i love this product':
    {'positive', 'negative'}
}
```

These should usually go to:

```text
Automatic detection
        ↓
Human review
        ↓
Correct / Remove
```

---

# 4. Method 3: Model confidence

Suppose you train a preliminary classifier.

For each example:

```text
Training Label: NEGATIVE

Model Prediction: POSITIVE
Confidence: 99%
```

This example may be mislabeled.

Example:

```python
prediction = {
    "predicted_label": "positive",
    "confidence": 0.99
}

true_label = "negative"

if (
    prediction["predicted_label"] != true_label
    and prediction["confidence"] > 0.95
):
    print("Potential noisy label")
```

But don't blindly delete it.

Why?

The model itself may be wrong.

Instead:

```text
High-confidence disagreement
        ↓
Flag
        ↓
Review
        ↓
Correct / Keep / Remove
```

---

# 5. Cross-validation based noise detection

A better technique is **out-of-fold prediction**.

Instead of checking examples using a model trained on the same example:

```text
Example
   ↓
Model trained on example
   ↓
Prediction
```

Use:

```text
Dataset
   │
   ├── Fold 1
   ├── Fold 2
   └── Fold 3

Train without Fold 1
       ↓
Predict Fold 1

Train without Fold 2
       ↓
Predict Fold 2
```

This is more reliable.

Example with scikit-learn:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
```

Dataset:

```python
texts = [
    "Amazing product",
    "Terrible service",
    "I love this",
    "I hate this",
    "Great experience",
    "Awful product"
]

labels = [
    "positive",
    "negative",
    "positive",
    "negative",
    "negative",  # possible noisy label
    "negative"
]
```

Vectorize:

```python
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
```

Cross-validation:

```python
skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

predictions = np.empty(
    len(labels),
    dtype=object
)

for train_idx, val_idx in skf.split(X, labels):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = np.array(labels)[train_idx]

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    predictions[val_idx] = model.predict(
        X_val
    )
```

Find disagreement:

```python
for text, true_label, predicted in zip(
    texts,
    labels,
    predictions
):

    if true_label != predicted:

        print(
            "Possible noisy example:"
        )

        print(
            "Text:",
            text
        )

        print(
            "Dataset label:",
            true_label
        )

        print(
            "Model prediction:",
            predicted
        )
```

---

# 6. LLM-as-a-judge for instruction fine-tuning

For LLM datasets, classification-style confidence isn't always enough.

You can evaluate examples like this:

```text
Question
    ↓
Candidate Answer
    ↓
Reference / Source
    ↓
LLM Judge
    ↓
Quality Score
```

Prompt:

```text
You are evaluating a fine-tuning example.

Question:
{question}

Answer:
{answer}

Evaluate:

1. Is the answer relevant?
2. Is the answer factually correct?
3. Is the answer complete?
4. Does the answer follow the instruction?

Return JSON:

{
    "relevance": 0-10,
    "correctness": 0-10,
    "completeness": 0-10,
    "quality": 0-10,
    "reason": "..."
}
```

Example:

```python
def should_keep(score):

    return (
        score["relevance"] >= 8
        and score["correctness"] >= 8
        and score["quality"] >= 8
    )
```

A production workflow:

```text
Fine-tuning example
        ↓
Automatic checks
        ↓
LLM judge
        ↓
High confidence?
    ┌───────┴────────┐
   Yes               No
    │                 │
Keep              Human Review
```

---

# 7. Use multiple annotators

If humans create labels, use agreement checking.

Example:

```text
Text:
"This is okay."

Annotator 1 → POSITIVE
Annotator 2 → NEUTRAL
Annotator 3 → NEUTRAL
```

Majority vote:

```text
NEUTRAL ✓
```

Code:

```python
from collections import Counter


def majority_vote(labels):

    counts = Counter(labels)

    return counts.most_common(1)[0][0]
```

Example:

```python
labels = [
    "positive",
    "neutral",
    "neutral"
]

print(
    majority_vote(labels)
)
```

Output:

```text
neutral
```

For better annotation analysis, measure inter-annotator agreement, such as **Cohen's kappa** for two annotators or **Fleiss' kappa** for multiple annotators.

---

# 8. Don't automatically remove every suspicious example

Use three buckets:

```text
Example
   │
   ├── High quality → Keep
   │
   ├── Uncertain → Human review
   │
   └── Clearly bad → Remove
```

This is usually safer than:

```text
Low model confidence
       ↓
Delete immediately ❌
```

---

# 9. Handling noisy labels during training

Sometimes you cannot clean everything.

Strategies include:

### A. Remove clearly bad examples

```python
cleaned_data = [
    x
    for x in dataset
    if x["quality_score"] >= 0.8
]
```

### B. Down-weight suspicious examples

Instead of:

[
Loss = \frac{1}{N}\sum L_i
]

Use weighted loss:

[
Loss =
\frac{
\sum w_i L_i
}{
\sum w_i
}
]

Where suspicious examples have smaller weights.

Example:

```python
import torch
import torch.nn.functional as F


loss_per_example = torch.tensor(
    [0.2, 0.5, 2.0, 0.1]
)

weights = torch.tensor(
    [1.0, 1.0, 0.3, 1.0]
)

weighted_loss = (
    loss_per_example * weights
).sum() / weights.sum()

print(weighted_loss)
```

The suspicious third example contributes less.

---

# 10. How do you handle class imbalance?

## What is class imbalance?

Suppose you have:

```text
Positive     9,000
Negative       800
Neutral        200
```

The model may learn:

```text
Predict POSITIVE
```

for almost everything.

It can achieve:

[
Accuracy = 90%
]

but still perform terribly for minority classes.

---

# 11. First detect imbalance

Code:

```python
from collections import Counter


labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "neutral"
]

counts = Counter(labels)

print(counts)
```

Output:

```text
Counter({
    'positive': 3,
    'negative': 1,
    'neutral': 1
})
```

Visual:

```text
Positive  ██████████████████
Negative  ██
Neutral   █
```

---

# 12. Method 1: Collect more minority examples

This is usually the best solution.

```text
Before:

Positive   9000
Negative    800
Neutral     200
```

Collect more:

```text
After:

Positive   9000
Negative   3000
Neutral    2500
```

Why is this better than duplication?

Because you get:

```text
More diversity
More real-world patterns
Less overfitting
```

---

# 13. Method 2: Oversampling

If collecting more data is difficult:

```text
Minority class
     ↓
Duplicate / resample
     ↓
More balanced dataset
```

Example:

```python
from sklearn.utils import resample


majority = [
    x for x in dataset
    if x["label"] == "positive"
]

minority = [
    x for x in dataset
    if x["label"] == "negative"
]
```

Oversample:

```python
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_dataset = (
    majority
    + minority_upsampled
)
```

Problem:

```text
Same examples repeated
        ↓
Overfitting risk
```

Better:

```text
Real minority examples
        +
Carefully validated augmentation
```

---

# 14. Method 3: Undersampling

Reduce the majority class.

```text
Before:

Positive   9000
Negative    800
Neutral     200
```

After:

```text
Positive   1000
Negative    800
Neutral     200
```

Code:

```python
majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=1000,
    random_state=42
)
```

Risk:

```text
Throwing away useful information
```

For large datasets, undersampling can be useful. For small datasets, it can be harmful.

---

# 15. Method 4: Class weights

Instead of changing the dataset, change the loss.

For cross-entropy:

[
Loss =
-\sum_c w_c y_c \log(p_c)
]

Where:

[
w_c
]

is higher for minority classes.

Example:

```text
Positive → weight 1

Negative → weight 5

Neutral → weight 20
```

Code:

```python
import torch
import torch.nn as nn


class_weights = torch.tensor(
    [
        1.0,
        5.0,
        20.0
    ]
)

loss_fn = nn.CrossEntropyLoss(
    weight=class_weights
)
```

Now mistakes on minority classes are penalized more heavily.

A common automatic formula is:

[
w_i =
\frac{N}
{K \cdot n_i}
]

Where:

* (N) = total examples
* (K) = number of classes
* (n_i) = examples in class (i)

Code:

```python
import numpy as np
from collections import Counter


labels = np.array([
    0, 0, 0, 0,
    1,
    2
])

counts = np.bincount(labels)

total = len(labels)

num_classes = len(counts)

weights = (
    total /
    (num_classes * counts)
)

print(weights)
```

The minority classes receive larger weights.

---

# 16. Method 5: Weighted sampling

Instead of oversampling the dataset permanently, sample minority examples more frequently.

PyTorch:

```python
from torch.utils.data import (
    WeightedRandomSampler
)
```

Example labels:

```python
labels = [0, 0, 0, 0, 1, 2]
```

Count:

```python
from collections import Counter


counts = Counter(labels)

class_weights = {
    label: 1 / count
    for label, count
    in counts.items()
}
```

Create sample weights:

```python
sample_weights = [
    class_weights[label]
    for label in labels
]
```

Sampler:

```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
```

Now minority examples have a higher probability of being selected.

---

# 17. How do you handle class imbalance in LLM fine-tuning?

Suppose you're fine-tuning an LLM to classify support tickets:

```text
Billing           50,000
Technical          30,000
Account            10,000
Security            2,000
```

A naive dataset may make the model weak at:

```text
Security tickets ❌
```

Possible approach:

```text
Original Data
      │
      ├── Keep high-quality majority examples
      │
      ├── Collect more real minority examples
      │
      ├── Generate diverse minority examples
      │
      ├── Validate synthetic examples
      │
      └── Control sampling during training
```

Example balancing:

```python
from collections import defaultdict
import random


by_class = defaultdict(list)

for example in dataset:

    by_class[
        example["label"]
    ].append(example)


for label, examples in by_class.items():

    print(
        label,
        len(examples)
    )
```

Controlled sampling:

```python
TARGET_PER_CLASS = 5000

balanced_dataset = []

for label, examples in by_class.items():

    if len(examples) >= TARGET_PER_CLASS:

        selected = random.sample(
            examples,
            TARGET_PER_CLASS
        )

    else:

        selected = random.choices(
            examples,
            k=TARGET_PER_CLASS
        )

    balanced_dataset.extend(
        selected
    )
```

But blindly duplicating minority examples is not ideal.

Better:

```text
2,000 real examples
      +
3,000 diverse validated examples
      ↓
Better minority coverage
```

---

# 18. Evaluate correctly

With imbalanced data, don't rely only on accuracy.

Example:

```text
100 examples

90 Positive
10 Negative
```

Model predicts:

```text
Everything → Positive
```

Accuracy:

```text
90%
```

But:

```text
Negative Recall = 0%
```

Use:

* Precision
* Recall
* F1-score
* Macro F1
* Per-class metrics
* Confusion matrix

Code:

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


y_true = [
    "positive",
    "positive",
    "negative",
    "negative"
]

y_pred = [
    "positive",
    "positive",
    "positive",
    "negative"
]

print(
    classification_report(
        y_true,
        y_pred
    )
)
```

For imbalanced datasets, **macro F1** is often more informative than accuracy because each class contributes equally.

---

# 19. Noisy labels + class imbalance together

This is a particularly dangerous situation.

Imagine:

```text
Class A → 100,000 examples
Class B → 1,000 examples
```

Then Class B contains:

```text
300 noisy examples
```

That means:

```text
30% of Class B is noisy
```

Simply oversampling Class B will also oversample bad labels.

Bad approach:

```text
Noisy minority class
        ↓
Oversample
        ↓
More noisy data ❌
```

Better:

```text
Detect noisy labels
        ↓
Clean minority data
        ↓
Human review
        ↓
Collect / generate quality examples
        ↓
Balance dataset
```

---

# 20. Production workflow

For a production ML/LLM pipeline:

```text
Raw Dataset
      │
      ▼
Schema Validation
      │
      ▼
Duplicate Detection
      │
      ▼
Noisy Label Detection
      │
      ├── Rules
      ├── Model disagreement
      ├── Cross-validation
      ├── LLM judge
      └── Human review
      │
      ▼
Clean Dataset
      │
      ▼
Check Class Distribution
      │
      ├── Collect minority data
      ├── Validated augmentation
      ├── Weighted sampling
      └── Class weights
      │
      ▼
Train Model
      │
      ▼
Per-class Evaluation
      │
      ▼
Production Monitoring
```

---

# Interview answer

## How do you handle noisy labels?

> I first try to prevent noisy labels through clear annotation guidelines and validation. For existing datasets, I detect suspicious examples using schema checks, duplicate and conflict detection, model disagreement, out-of-fold predictions, and for LLM datasets, reference-based evaluation or LLM-as-a-judge. I don't automatically delete every suspicious example; I use confidence thresholds and send ambiguous cases to human review. Clearly bad examples are removed or corrected, while uncertain examples can be down-weighted during training.

## How do you handle class imbalance?

> First, I measure the class distribution and evaluate per-class metrics rather than relying on accuracy. My preferred solution is collecting more high-quality examples for minority classes. If that isn't possible, I use controlled oversampling, weighted sampling, or class-weighted loss depending on the model and task. I evaluate with macro F1, precision, recall, and a confusion matrix. For LLM fine-tuning, I also ensure minority intents are represented by diverse examples rather than simply duplicating the same data.

### Key interview insight

```text
Noisy labels:
Improve data quality
        ↓
Correct / Remove / Down-weight

Class imbalance:
Improve data representation
        ↓
Collect / Balance / Re-weight
```

The important production principle is:

> **Never blindly oversample a noisy minority class, because you amplify both the signal and the noise.**
