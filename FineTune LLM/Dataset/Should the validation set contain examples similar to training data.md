# Should the validation set contain examples similar to training data?

## Short answer

**Yes — similar in distribution, but not the same or leaked examples.**

This distinction is extremely important.

```text
Training Data                    Validation Data

Same problem domain        →     Same problem domain       ✅
Same data distribution     →     Same data distribution    ✅
Similar task               →     Similar task              ✅

Exact duplicates           →     Exact duplicates          ❌
Same conversation          →     Same conversation         ❌
Same document chunks       →     Same document             ❌
Near-identical examples    →     Near-identical examples   ⚠️ Usually avoid
```

The validation set should answer:

> **"Can the model generalize to new examples that look like the kind of data it will see in production?"**

---

# 1. What does "similar" actually mean?

Suppose you're training a model to classify customer support tickets.

## Training data

```text
"My payment failed"              → Billing
"I was charged twice"            → Billing
"Cannot log into my account"     → Account
"My password reset failed"       → Account
```

## Good validation data

```text
"My credit card payment failed"  → Billing
"Why was I billed two times?"    → Billing
"I cannot access my profile"     → Account
"Password recovery is not working" → Account
```

These are **similar in task and domain**, but are new examples.

```text
Training
   ↓
Payment issues
Login issues
Billing issues
   ↓
Model learns patterns
   ↓
Validation
   ↓
New payment/login/billing examples
```

This is what you want.

---

# 2. Bad validation data: exact duplicate

## Training

```text
What is LoRA?

LoRA is a parameter-efficient fine-tuning technique.
```

## Validation

```text
What is LoRA?

LoRA is a parameter-efficient fine-tuning technique.
```

This is leakage.

The validation score might look excellent:

```text
Validation Accuracy = 98%
```

But the model may simply be memorizing.

---

# 3. Good validation data for LLM fine-tuning

Suppose you are fine-tuning an LLM for AI/ML interview questions.

## Training

```json
{
    "instruction": "What is LoRA?",
    "response": "LoRA is a parameter-efficient fine-tuning technique that uses low-rank matrices."
}
```

```json
{
    "instruction": "What is QLoRA?",
    "response": "QLoRA combines 4-bit quantization with LoRA adapters."
}
```

## Validation

```json
{
    "instruction": "Why does LoRA use low-rank matrices?",
    "response": "LoRA assumes the required parameter updates can often be represented in a lower-dimensional space."
}
```

```json
{
    "instruction": "How does QLoRA reduce memory usage?",
    "response": "QLoRA keeps the base model quantized while training small LoRA adapters."
}
```

These are related concepts:

```text
Train:
LoRA
QLoRA

Validation:
LoRA low-rank reasoning
QLoRA memory reasoning
```

That is generally good.

---

# 4. The key concept: IID

For many ML problems, we want training and validation data to approximately follow the same distribution.

This is often described as:

[
P_{train}(X, Y) \approx P_{validation}(X, Y)
]

Meaning:

```text
Training distribution
        ≈
Validation distribution
```

Example:

```text
Training:

Billing       40%
Technical     30%
Account       20%
Security      10%
```

Validation:

```text
Billing       41%
Technical     29%
Account       20%
Security      10%
```

Good.

But:

```text
Training:

Billing       40%
Technical     30%
Account       20%
Security      10%
```

Validation:

```text
Billing       0%
Technical     0%
Account       0%
Security    100%
```

This evaluates something different.

Unless your actual production goal is to test domain shift.

---

# 5. Example with code: similar distribution

Let's create a dataset.

```python
from collections import Counter

dataset = [
    {"text": "Payment failed", "label": "billing"},
    {"text": "Charged twice", "label": "billing"},
    {"text": "Refund not received", "label": "billing"},

    {"text": "App crashed", "label": "technical"},
    {"text": "Page not loading", "label": "technical"},
    {"text": "API returns error", "label": "technical"},

    {"text": "Cannot login", "label": "account"},
    {"text": "Password reset failed", "label": "account"},
    {"text": "Account locked", "label": "account"},
]
```

Split using stratification:

```python
from sklearn.model_selection import train_test_split


texts = [
    item["text"]
    for item in dataset
]

labels = [
    item["label"]
    for item in dataset
]


X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
```

Check distributions:

```python
print(
    "Train:",
    Counter(y_train)
)

print(
    "Validation:",
    Counter(y_val)
)
```

The idea is:

```text
Train:
billing      ~33%
technical    ~33%
account      ~33%

Validation:
billing      ~33%
technical    ~33%
account      ~33%
```

Thus, validation is representative of the same production task.

> Note: stratification can fail or be unstable with extremely small datasets because each class needs enough examples to appear in each split.

---

# 6. How do you check whether distributions are similar?

For classification, compare class distributions.

```python
from collections import Counter


def get_distribution(labels):

    counts = Counter(labels)

    total = len(labels)

    return {
        label: round(
            count / total,
            3
        )
        for label, count in counts.items()
    }


train_distribution = get_distribution(
    y_train
)

val_distribution = get_distribution(
    y_val
)

print("Train:", train_distribution)
print("Validation:", val_distribution)
```

Example:

```text
Train:
{
    'billing': 0.40,
    'technical': 0.30,
    'account': 0.30
}

Validation:
{
    'billing': 0.42,
    'technical': 0.29,
    'account': 0.29
}
```

That is reasonably similar.

---

# 7. Similar does NOT mean duplicate

Let's visualize the difference.

## Good

```text
TRAIN

"What is LoRA?"
"How does LoRA work?"
"Explain LoRA rank."


VALIDATION

"Why does LoRA use low-rank matrices?"
"When should LoRA be used?"
"How does LoRA reduce trainable parameters?"
```

Same topic:

```text
LoRA
```

Different examples:

```text
New wording
New questions
New reasoning
```

Good.

---

## Bad

```text
TRAIN

"What is LoRA?"

VALIDATION

"What is LoRA?"
```

Exact duplicate.

---

## Potentially bad

```text
TRAIN:

"What is LoRA?"

VALIDATION:

"Explain what LoRA is."
```

This is a near duplicate.

Whether you remove it depends on your evaluation goal.

For a strict generalization benchmark, remove it.

For a production workload where users frequently paraphrase the same questions, you may want a separate **paraphrase robustness test**.

---

# 8. Code to detect exact overlap

```python
def normalize(text):

    return (
        text
        .lower()
        .strip()
    )


train_set = {
    normalize(text)
    for text in X_train
}

val_set = {
    normalize(text)
    for text in X_val
}


overlap = (
    train_set
    & val_set
)

print(
    "Exact overlap:",
    overlap
)

assert len(overlap) == 0
```

Expected:

```text
Exact overlap: set()
```

---

# 9. Code to detect near duplicates

Exact duplicates are easy.

Near duplicates are harder.

Example:

```text
Training:
What is LoRA?

Validation:
Can you explain LoRA?
```

We can use embeddings.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
```

Load an embedding model:

```python
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

Generate embeddings:

```python
train_embeddings = model.encode(
    X_train,
    normalize_embeddings=True
)

val_embeddings = model.encode(
    X_val,
    normalize_embeddings=True
)
```

Calculate similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity


similarity_matrix = cosine_similarity(
    val_embeddings,
    train_embeddings
)
```

Find closest training example for each validation example:

```python
threshold = 0.95


for val_index, similarities in enumerate(
    similarity_matrix
):

    best_train_index = similarities.argmax()

    similarity_score = (
        similarities[best_train_index]
    )

    if similarity_score > threshold:

        print(
            "\nPossible near duplicate"
        )

        print(
            "Validation:",
            X_val[val_index]
        )

        print(
            "Training:",
            X_train[best_train_index]
        )

        print(
            "Similarity:",
            similarity_score
        )
```

Example output:

```text
Possible near duplicate

Validation:
Can you explain LoRA?

Training:
What is LoRA?

Similarity:
0.97
```

Then decide:

```text
Keep?
Remove?
Move to a separate robustness benchmark?
```

---

# 10. LLM-specific example

Suppose you're instruction-tuning a model.

Your dataset:

```python
dataset = [
    {
        "topic": "lora",
        "instruction": "What is LoRA?",
        "response": "LoRA is a parameter-efficient fine-tuning technique."
    },
    {
        "topic": "lora",
        "instruction": "How does LoRA reduce memory?",
        "response": "It trains small low-rank matrices instead of all model parameters."
    },
    {
        "topic": "qlora",
        "instruction": "What is QLoRA?",
        "response": "QLoRA uses a quantized base model and trainable LoRA adapters."
    },
    {
        "topic": "qlora",
        "instruction": "Why does QLoRA use 4-bit quantization?",
        "response": "It reduces GPU memory requirements."
    }
]
```

A good split should not necessarily isolate entire topics.

For example:

```text
Training:
LoRA → What is LoRA?
LoRA → How does LoRA work?

Validation:
LoRA → Why is LoRA parameter efficient?
LoRA → What happens when rank increases?
```

This tests whether the model generalizes **within the same task domain**.

That is often exactly what you want.

---

# 11. But when should you split entire topics?

Sometimes you want a harder test.

For example:

```text
Training topics:

LoRA
QLoRA
RAG
Embeddings


Validation topics:

LangGraph
MCP
Multi-Agent Systems
```

Now you are testing:

```text
Can the model generalize to unseen topics?
```

This is an **out-of-distribution (OOD)** or **domain generalization** evaluation.

It answers a different question.

---

# 12. Different evaluation strategies

You can have multiple validation sets.

## Validation Set A: In-distribution

```text
Training:
Support tickets from common categories

Validation:
New support tickets from same categories
```

Tests:

```text
Normal production performance
```

---

## Validation Set B: Paraphrase robustness

```text
Training:
"What is LoRA?"

Validation:
"Can you explain LoRA?"
```

Tests:

```text
Robustness to wording variation
```

---

## Validation Set C: OOD

```text
Training:
AI interview questions

Validation:
Legal questions
```

Tests:

```text
Domain generalization
```

---

## Validation Set D: Edge cases

```text
Very long input
Typos
Ambiguous questions
Multiple instructions
Rare categories
```

Tests:

```text
Production robustness
```

A mature ML system often uses several evaluation sets:

```text
                    ┌── In-distribution
                    │
Model ──► Evaluation ├── Paraphrase
                    │
                    ├── Edge cases
                    │
                    └── Out-of-distribution
```

---

# 13. Example: Production-quality split strategy

Suppose you have 100,000 customer-support conversations.

```text
100,000 conversations
```

A reasonable strategy:

```text
80% → Training

10% → Validation

10% → Test
```

But don't randomly split individual messages.

Instead:

```text
Conversation 123
    ├── Message 1
    ├── Message 2
    ├── Message 3
    └── Message 4

Entire conversation
        ↓
One split only
```

Code:

```python
from sklearn.model_selection import GroupShuffleSplit


groups = [
    example["conversation_id"]
    for example in dataset
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)


train_idx, temp_idx = next(
    splitter.split(
        dataset,
        groups=groups
    )
)
```

Then split the temporary set:

```python
train_data = [
    dataset[i]
    for i in train_idx
]

temp_data = [
    dataset[i]
    for i in temp_idx
]
```

Now:

```text
Train: 80%
Temporary: 20%
```

Split temporary:

```python
temp_groups = [
    item["conversation_id"]
    for item in temp_data
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42
)


val_idx, test_idx = next(
    splitter.split(
        temp_data,
        groups=temp_groups
    )
)


val_data = [
    temp_data[i]
    for i in val_idx
]

test_data = [
    temp_data[i]
    for i in test_idx
]
```

Verify:

```python
train_ids = {
    x["conversation_id"]
    for x in train_data
}

val_ids = {
    x["conversation_id"]
    for x in val_data
}

test_ids = {
    x["conversation_id"]
    for x in test_data
}


assert not train_ids & val_ids
assert not train_ids & test_ids
assert not val_ids & test_ids
```

This prevents conversation-level leakage.

---

# 14. A practical LLM fine-tuning example

Suppose you fine-tune a model to answer questions about FastAPI.

### Training

```text
What is FastAPI?
How do dependencies work?
What is Depends()?
How does async work?
How do I create a router?
```

### Validation

```text
How does dependency injection work in FastAPI?
When should I use async endpoints?
How should I organize FastAPI routers?
```

This is good because:

```text
Same domain        ✓
Same task          ✓
Same style         ✓
New questions      ✓
No duplicate       ✓
```

---

# 15. What happens if validation is too different?

Suppose:

```text
Training:

AI
ML
LLM
RAG
```

Validation:

```text
Medical diagnosis
Legal contracts
Financial auditing
```

A poor score might not mean your fine-tuning failed.

It may simply mean:

[
P_{train}(X) \neq P_{validation}(X)
]

This is called **distribution shift**.

Your validation set is now measuring something different from your primary deployment scenario.

---

# 16. What happens if validation is too similar?

Suppose:

```text
Training:
How does LoRA work?

Validation:
How does LoRA work?

Test:
How does LoRA work?
```

Result:

```text
Validation Score = 99%
```

But this may represent:

```text
Memorization
```

rather than:

```text
Generalization
```

---

# 17. The ideal relationship

Think of it like this:

```text
              SAME DISTRIBUTION
                    │
                    │
        ┌───────────┴───────────┐
        │                       │
Training Examples         Validation Examples
        │                       │
        │                       │
Seen by model          Never seen by model
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
             Measure generalization
```

The key is:

> **Same population, different samples.**

---

# 18. Interview answer

### Should the validation set contain examples similar to the training set?

A strong answer is:

> Yes, the validation set should generally come from the same distribution as the training and expected production data, but it must contain examples the model has never seen during training. Similarity should exist at the task, domain, label distribution, and data characteristics level, not through duplicated or leaked examples. For example, if I'm training an LLM on FastAPI questions, both training and validation can contain FastAPI questions, but the validation questions should be new and ideally test different wording or reasoning paths.

### What would you check?

> I check exact duplicates, semantic near duplicates, group overlap such as users, documents, or conversations, class distribution, and time boundaries. I use stratified splitting for classification and group-based splitting when related examples must stay together.

---

# Final rule to remember

```text
VALIDATION SHOULD BE:

Similar enough
    ↓
to represent production

Different enough
    ↓
to test generalization
```

Or in one sentence:

> **The validation set should be drawn from the same data distribution, but every validation example should represent genuinely unseen information for the model.**
