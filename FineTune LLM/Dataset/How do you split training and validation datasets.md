# How do you split training and validation datasets?

This is a very important question for ML and LLM fine-tuning.

The goal is:

```text
Raw Dataset
    │
    ├── Training Set ─────► Model learns from this
    │
    ├── Validation Set ───► Tune hyperparameters / choose checkpoint
    │
    └── Test Set ─────────► Final unbiased evaluation
```

A common split is:

```text
Train        80%
Validation   10%
Test         10%
```

For example, with 10,000 examples:

```text
Train       8,000
Validation  1,000
Test        1,000
```

The exact ratio depends on dataset size.

---

# 1. What is the purpose of each split?

## Training dataset

The model updates its weights using this data.

```text
Training Example
      ↓
Forward Pass
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Update Weights
```

Example:

```python
loss.backward()
optimizer.step()
```

---

## Validation dataset

The validation data is **not used to update model weights**.

It is used to:

* choose learning rate
* choose number of epochs
* compare models
* detect overfitting
* choose the best checkpoint

Example:

```text
Epoch 1

Train Loss:       1.20
Validation Loss:  1.30


Epoch 5

Train Loss:       0.30
Validation Loss:  0.60


Epoch 20

Train Loss:       0.05
Validation Loss:  2.00
                         ↑
                    Overfitting
```

---

## Test dataset

The test set should be used only for the final evaluation.

```text
Train
  ↓
Train model

Validation
  ↓
Tune model

Test
  ↓
Final evaluation
```

A common mistake is:

```text
Test result
   ↓
Change model
   ↓
Test again
   ↓
Change model
   ↓
Test again
```

Eventually, you are indirectly training for the test set.

That is a form of **evaluation leakage**.

---

# 2. Basic dataset split with scikit-learn

Suppose:

```python
dataset = [
    {
        "instruction": "What is LoRA?",
        "response": "LoRA is a parameter-efficient fine-tuning technique."
    },
    {
        "instruction": "What is QLoRA?",
        "response": "QLoRA combines quantization with LoRA."
    },
    # many more examples...
]
```

Use `train_test_split`.

```python
from sklearn.model_selection import train_test_split


train_data, temp_data = train_test_split(
    dataset,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

validation_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    shuffle=True
)
```

Result:

```text
80% → Train
10% → Validation
10% → Test
```

Check:

```python
print("Train:", len(train_data))
print("Validation:", len(validation_data))
print("Test:", len(test_data))
```

---

# 3. Using Hugging Face `datasets`

For LLM fine-tuning, this is common.

Install:

```bash
pip install datasets
```

Example:

```python
from datasets import Dataset


dataset = Dataset.from_list([
    {
        "instruction": "What is LoRA?",
        "response": "LoRA is a parameter-efficient fine-tuning method."
    },
    {
        "instruction": "What is RAG?",
        "response": "RAG retrieves relevant information before generating an answer."
    },
])
```

Split:

```python
train_test = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    shuffle=True
)
```

Now:

```python
train_dataset = train_test["train"]
temp_dataset = train_test["test"]
```

Split the remaining 20%:

```python
validation_test = temp_dataset.train_test_split(
    test_size=0.5,
    seed=42,
    shuffle=True
)

validation_dataset = validation_test["train"]
test_dataset = validation_test["test"]
```

Final:

```text
Train        80%
Validation   10%
Test         10%
```

---

# 4. Why `random_state` / `seed` is important

Consider:

```python
train_test_split(dataset)
```

Every execution may create a different split.

That makes experiments hard to reproduce.

Use:

```python
random_state=42
```

or:

```python
seed=42
```

Example:

```python
train_data, val_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42
)
```

Now your experiment is reproducible.

In production, store:

```text
Dataset Version: 1.2
Split Seed: 42
Train Examples: 80,000
Validation Examples: 10,000
Test Examples: 10,000
```

---

# 5. How do you split classification data?

Suppose:

```text
Positive → 8,000
Negative → 1,500
Neutral  → 500
```

A random split could accidentally produce:

```text
Training:
Positive → 6,500
Negative → 1,450
Neutral  → 50
```

Validation:

```text
Positive → 1,500
Negative → 50
Neutral  → 450
```

This is bad.

Use **stratified splitting**.

---

# 6. Stratified split

```text
Full Dataset

Positive  80%
Negative  15%
Neutral    5%
      │
      ▼

Train

Positive  80%
Negative  15%
Neutral    5%

Validation

Positive  80%
Negative  15%
Neutral    5%
```

Code:

```python
from sklearn.model_selection import train_test_split


X = [
    "Great product",
    "Bad product",
    "Average product",
    # ...
]

y = [
    "positive",
    "negative",
    "neutral",
    # ...
]


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Then:

```python
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
```

`stratify=y` preserves the class distribution.

---

# 7. How do you split conversational fine-tuning data?

Suppose:

```python
example = {
    "conversation_id": "conv_123",
    "messages": [
        {
            "role": "user",
            "content": "My payment failed"
        },
        {
            "role": "assistant",
            "content": "Can you provide the error message?"
        },
        {
            "role": "user",
            "content": "I see error 101"
        },
        {
            "role": "assistant",
            "content": "Error 101 means..."
        }
    ]
}
```

You should usually keep the **entire conversation together**.

Bad:

```text
Train:
User: My payment failed
Assistant: Can you provide the error?

Validation:
User: I see error 101
Assistant: Error 101 means...
```

This leaks conversation context.

Instead:

```text
Conversation 123
       │
       ▼
Entire conversation → Train
```

or:

```text
Entire conversation → Validation
```

---

# 8. Group-based splitting

Use `GroupShuffleSplit`.

```python
from sklearn.model_selection import GroupShuffleSplit
```

Example:

```python
data = [
    {
        "conversation_id": "conv_1",
        "text": "Question 1",
        "label": "A"
    },
    {
        "conversation_id": "conv_1",
        "text": "Question 2",
        "label": "A"
    },
    {
        "conversation_id": "conv_2",
        "text": "Question 3",
        "label": "B"
    }
]
```

Groups:

```python
groups = [
    item["conversation_id"]
    for item in data
]
```

Split:

```python
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        data,
        groups=groups
    )
)
```

Create datasets:

```python
train_data = [
    data[i]
    for i in train_idx
]

validation_data = [
    data[i]
    for i in val_idx
]
```

Verify:

```python
train_groups = {
    item["conversation_id"]
    for item in train_data
}

validation_groups = {
    item["conversation_id"]
    for item in validation_data
}

assert (
    train_groups
    .intersection(validation_groups)
    == set()
)
```

This is very important for:

* conversations
* users
* customers
* documents
* companies
* sessions

---

# 9. What is data leakage?

Data leakage means:

> Information that should not be available during training accidentally influences the model or evaluation.

This causes:

```text
Offline evaluation → Excellent
Production         → Poor
```

because the evaluation was unrealistic.

---

# 10. Type 1: Exact duplicate leakage

Example:

```text
Training:
What is LoRA?
→ LoRA uses low-rank matrices.

Validation:
What is LoRA?
→ LoRA uses low-rank matrices.
```

The model has already seen the exact example.

Your validation score becomes artificially high.

---

## Detect overlap

Create a hash:

```python
import hashlib
import json


def hash_example(example):

    text = json.dumps(
        example,
        sort_keys=True
    )

    return hashlib.sha256(
        text.encode()
    ).hexdigest()
```

Check overlap:

```python
train_hashes = {
    hash_example(x)
    for x in train_data
}

val_hashes = {
    hash_example(x)
    for x in validation_data
}

overlap = (
    train_hashes
    .intersection(val_hashes)
)

print(
    "Overlapping examples:",
    len(overlap)
)
```

You want:

```text
Overlapping examples: 0
```

---

# 11. Near-duplicate leakage

Exact duplicates are easy.

But consider:

### Training

```text
Explain LoRA.

LoRA is a fine-tuning technique using low-rank matrices.
```

### Validation

```text
What does LoRA mean?

LoRA is a parameter-efficient method that trains low-rank adapters.
```

These are different strings but almost identical semantically.

If your goal is measuring generalization, these may create leakage.

---

## Detect semantic overlap

Use embeddings.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
```

Load model:

```python
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

Convert examples to text:

```python
def example_to_text(example):

    return (
        example["instruction"]
        + "\n"
        + example["response"]
    )
```

Generate embeddings:

```python
train_texts = [
    example_to_text(x)
    for x in train_data
]

val_texts = [
    example_to_text(x)
    for x in validation_data
]

train_embeddings = model.encode(
    train_texts,
    normalize_embeddings=True
)

val_embeddings = model.encode(
    val_texts,
    normalize_embeddings=True
)
```

Check similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(
    val_embeddings,
    train_embeddings
)
```

Find suspicious validation examples:

```python
threshold = 0.95

for val_index, row in enumerate(similarity):

    max_similarity = row.max()

    if max_similarity > threshold:

        train_index = row.argmax()

        print(
            "Possible leakage"
        )

        print(
            "Validation:",
            val_texts[val_index]
        )

        print(
            "Training:",
            train_texts[train_index]
        )

        print(
            "Similarity:",
            max_similarity
        )
```

For large datasets, use FAISS or another ANN index rather than building a full similarity matrix.

---

# 12. Type 2: User leakage

Suppose you're training a customer-support model.

```text
Customer A:
Train → 50 conversations

Customer A:
Validation → 10 conversations
```

The model may learn customer-specific patterns.

This gives overly optimistic results.

Correct:

```text
Customer A → Train only
Customer B → Validation only
Customer C → Test only
```

Code:

```python
from sklearn.model_selection import GroupShuffleSplit


groups = [
    item["customer_id"]
    for item in dataset
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        dataset,
        groups=groups
    )
)
```

Verify:

```python
train_customers = {
    dataset[i]["customer_id"]
    for i in train_idx
}

val_customers = {
    dataset[i]["customer_id"]
    for i in val_idx
}

assert not (
    train_customers
    & val_customers
)
```

---

# 13. Type 3: Document leakage

This is especially important for LLMs.

Suppose you have one document:

```text
Employee Handbook
```

You chunk it:

```text
Chunk 1 → Train
Chunk 2 → Train
Chunk 3 → Validation ❌
Chunk 4 → Validation ❌
```

The chunks are strongly related.

The model may effectively memorize document-specific language.

Correct approach:

```text
Document A → Train
Document B → Validation
Document C → Test
```

Never split at the chunk level if you want document-level generalization.

---

## Code: group by document

```python
groups = [
    example["document_id"]
    for example in dataset
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        dataset,
        groups=groups
    )
)
```

Check:

```python
train_docs = {
    dataset[i]["document_id"]
    for i in train_idx
}

val_docs = {
    dataset[i]["document_id"]
    for i in val_idx
}

assert len(
    train_docs & val_docs
) == 0
```

---

# 14. Type 4: Time leakage

Suppose you predict future events.

Bad split:

```text
Random Split

Train:
January
March
June
August

Validation:
February
April
September
```

The training data can contain future information relative to validation.

Correct:

```text
Train:
Jan → June

Validation:
July → August

Test:
September → October
```

Code:

```python
dataset = sorted(
    dataset,
    key=lambda x: x["timestamp"]
)

n = len(dataset)

train_end = int(
    n * 0.8
)

val_end = int(
    n * 0.9
)

train_data = dataset[
    :train_end
]

validation_data = dataset[
    train_end:val_end
]

test_data = dataset[
    val_end:
]
```

This is important for:

* financial models
* forecasting
* fraud detection
* recommendation systems
* changing enterprise policies
* time-sensitive knowledge

---

# 15. Type 5: Feature leakage

Suppose you're predicting:

```text
Will customer churn?
```

Input:

```text
Customer usage
Login frequency
Purchase history
```

Then accidentally include:

```text
account_closed_date
```

This feature may only be known **after** the customer churns.

The model will appear extremely accurate.

But it cannot use that feature in production.

Example:

```text
Input Features

login_count                 ✓
purchase_count              ✓
account_closed_date         ❌
```

Rule:

> Only use information available at prediction time.

---

# 16. Type 6: Target leakage

Example:

```text
Input:
"The answer is positive"

Label:
POSITIVE
```

The target is directly or indirectly present in the input.

Another example:

```text
Text:
Customer sentiment: NEGATIVE
Customer says: I hate this product.

Label:
NEGATIVE
```

The model can simply copy the answer.

You should remove target-bearing fields.

```python
LEAKY_FIELDS = {
    "label",
    "target",
    "final_answer",
    "resolution"
}


def remove_leaky_fields(example):

    return {
        key: value
        for key, value
        in example.items()
        if key not in LEAKY_FIELDS
    }
```

Be careful: in instruction fine-tuning, the target answer must of course be retained separately as the training label, not included inside the model's input prompt.

---

# 17. A production-safe splitting function

Let's build a reusable splitter.

```python
from sklearn.model_selection import GroupShuffleSplit


def group_train_val_test_split(
    dataset,
    group_key,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
):

    assert (
        train_ratio
        + val_ratio
        + test_ratio
        == 1.0
    )

    groups = [
        item[group_key]
        for item in dataset
    ]

    # Step 1:
    # Train vs remaining
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio + test_ratio,
        random_state=random_state
    )

    train_idx, remaining_idx = next(
        splitter_1.split(
            dataset,
            groups=groups
        )
    )

    train_data = [
        dataset[i]
        for i in train_idx
    ]

    remaining_data = [
        dataset[i]
        for i in remaining_idx
    ]

    # Step 2:
    # Validation vs test
    remaining_groups = [
        item[group_key]
        for item in remaining_data
    ]

    test_fraction = (
        test_ratio
        / (val_ratio + test_ratio)
    )

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_state
    )

    val_idx, test_idx = next(
        splitter_2.split(
            remaining_data,
            groups=remaining_groups
        )
    )

    validation_data = [
        remaining_data[i]
        for i in val_idx
    ]

    test_data = [
        remaining_data[i]
        for i in test_idx
    ]

    return (
        train_data,
        validation_data,
        test_data
    )
```

Usage:

```python
train_data, val_data, test_data = (
    group_train_val_test_split(
        dataset=dataset,
        group_key="document_id"
    )
)
```

---

# 18. Verify your splits

Never assume the split is correct.

## Check sizes

```python
print(
    "Train:",
    len(train_data)
)

print(
    "Validation:",
    len(val_data)
)

print(
    "Test:",
    len(test_data)
)
```

---

## Check group overlap

```python
def get_groups(
    data,
    group_key
):

    return {
        x[group_key]
        for x in data
    }


train_groups = get_groups(
    train_data,
    "document_id"
)

val_groups = get_groups(
    val_data,
    "document_id"
)

test_groups = get_groups(
    test_data,
    "document_id"
)


assert not (
    train_groups
    & val_groups
)

assert not (
    train_groups
    & test_groups
)

assert not (
    val_groups
    & test_groups
)
```

---

# 19. Full example for an LLM fine-tuning dataset

Suppose:

```python
dataset = [
    {
        "document_id": "doc_1",
        "messages": [
            {
                "role": "user",
                "content": "What is LoRA?"
            },
            {
                "role": "assistant",
                "content": (
                    "LoRA is a parameter-efficient "
                    "fine-tuning method."
                )
            }
        ]
    },
    {
        "document_id": "doc_1",
        "messages": [
            {
                "role": "user",
                "content": "How does LoRA work?"
            },
            {
                "role": "assistant",
                "content": (
                    "LoRA trains low-rank matrices "
                    "while freezing the base model."
                )
            }
        ]
    },
    {
        "document_id": "doc_2",
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?"
            },
            {
                "role": "assistant",
                "content": (
                    "RAG retrieves relevant documents "
                    "before generating an answer."
                )
            }
        ]
    }
]
```

Split by `document_id`:

```python
train_data, val_data, test_data = (
    group_train_val_test_split(
        dataset,
        group_key="document_id"
    )
)
```

Result:

```text
Train:
doc_1

Validation:
doc_2

Test:
doc_3
```

This is safer than:

```text
Train:
doc_1 chunk 1
doc_1 chunk 2

Validation:
doc_1 chunk 3 ❌
```

---

# 20. Important issue: splitting before or after preprocessing?

A very common mistake:

```text
Full Dataset
     ↓
Fit preprocessing statistics
     ↓
Normalize all data
     ↓
Split
```

This can leak information.

Correct:

```text
Raw Dataset
     ↓
Split
     │
     ├── Train
     │     ↓
     │   Fit preprocessing
     │
     ├── Validation
     │     ↓
     │   Transform only
     │
     └── Test
           ↓
         Transform only
```

---

## Example: StandardScaler leakage

Wrong:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

# ❌ Fits on entire dataset
X_scaled = scaler.fit_transform(
    X
)

# Split after fitting
```

Correct:

```python
from sklearn.preprocessing import StandardScaler


# Split first
X_train, X_val = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

# Fit only on training data
X_train = scaler.fit_transform(
    X_train
)

# Validation only transforms
X_val = scaler.transform(
    X_val
)
```

The same principle applies to:

* feature scaling
* token vocabulary creation
* label encoders
* feature selection
* PCA
* imputation
* normalization statistics

---

# 21. LLM-specific leakage: prompt formatting

For instruction fine-tuning:

```text
Prompt:
Question: What is LoRA?

Answer:
```

The answer should be the target.

Correct conceptual training input:

```text
[USER]
What is LoRA?

[ASSISTANT]
LoRA is a parameter-efficient fine-tuning method.
```

During causal language-model training, the sequence can contain both prompt and answer, but you should usually compute the supervised loss only on the assistant response if your training objective is assistant-response generation.

For example:

```python
labels = input_ids.clone()

# Conceptually mask prompt tokens
labels[:prompt_length] = -100
```

`-100` means:

```text
Ignore these tokens when calculating loss.
```

So:

```text
Prompt tokens
Loss = ignored

Assistant tokens
Loss = calculated
```

This is not exactly the same as train/validation leakage, but it prevents accidentally training the model with an unintended objective.

---

# 22. Best practices for production

## Before splitting

```text
✓ Define the prediction/generalization unit
✓ Remove corrupt examples
✓ Remove exact duplicates
✓ Identify groups
✓ Identify time boundaries
✓ Define leakage rules
```

## During splitting

```text
✓ Fixed random seed
✓ Stratify when needed
✓ Group related examples
✓ Keep documents/conversations/users together
✓ Use chronological split for time-dependent data
```

## After splitting

```text
✓ Check exact duplicates
✓ Check semantic duplicates
✓ Check group overlap
✓ Check class distribution
✓ Check time boundaries
✓ Version the split
```

---

# Interview answer

### How do you split training and validation datasets?

> I usually split the data before any preprocessing that learns statistics from the dataset. For a standard classification problem, I use a reproducible random split, typically 80/10/10 for train, validation, and test, and use stratification when class distribution matters. For LLM datasets, I first identify the unit of independence. For example, all examples from the same conversation, user, document, or source should usually remain in the same split. I use group-based splitting to enforce this. For temporal problems, I use chronological splitting instead of random splitting.

### How do you prevent data leakage?

> I check leakage at multiple levels. I remove exact and near duplicates across splits, ensure related groups such as users, conversations, or documents don't appear in multiple splits, and use time-based splits when future information could leak into training. I also fit preprocessing transformations only on the training set and apply them to validation and test sets. Finally, I verify the split programmatically by checking group intersections, duplicate hashes, class distributions, and temporal boundaries.

## The key principle

```text
Validation data should simulate
data the model has never truly seen before.
```

If validation data is too similar to training data:

```text
Offline score: 95%
Production:    60%
```

A lower but realistic validation score is much more valuable than an artificially high score caused by leakage.
