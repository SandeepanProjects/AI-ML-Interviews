# How would you detect data leakage?

Data leakage means **information that should not be available during training or prediction accidentally influences the model**.

For LLM fine-tuning, leakage commonly happens when:

* The same or near-duplicate examples appear in train and validation/test sets.
* Data from the future is used to predict the past.
* The answer accidentally appears in the input.
* Preprocessing is fit on the entire dataset before splitting.
* Evaluation benchmark questions or answers appear in training data.
* Customer/session/document-level records are split incorrectly.

A strong interview answer:

> **I detect data leakage at multiple levels: schema and feature review, exact duplicate detection, near-duplicate semantic matching, group-level split validation, temporal validation, train-test preprocessing isolation, and benchmark contamination checks. I also build automated leakage checks into the dataset pipeline so a dataset cannot be promoted to training if leakage thresholds are exceeded.**

---

# 1. First understand the types of leakage

## A. Train-validation overlap

Example:

```text
Train:
Question: What is LoRA?
Answer: LoRA is a parameter-efficient fine-tuning technique.

Validation:
Question: What is LoRA?
Answer: LoRA is a parameter-efficient fine-tuning technique.
```

The model has effectively already seen the validation example.

---

## B. Near-duplicate leakage

```text
Train:
Explain how LoRA reduces trainable parameters.

Validation:
How does LoRA reduce the number of parameters that need training?
```

These are not exact duplicates, but may be semantically almost identical.

---

## C. Answer leakage

Example:

```text
Input:
Question: What is the capital of France?

Context:
The capital of France is Paris.

Target:
Paris
```

If your task is supposed to test knowledge without that context, the answer is leaking into the input.

---

## D. Temporal leakage

Suppose you predict:

```text
Customer churn in January
```

but train using:

```text
Account status updated in February
```

The model is using future information.

---

## E. Group leakage

Suppose one customer has 100 conversations.

Bad split:

```text
Customer A conversation 1 → Train
Customer A conversation 2 → Train
Customer A conversation 99 → Validation
```

The model may learn customer-specific patterns.

Better:

```text
Customer A → Train only

Customer B → Validation only
```

---

# 2. Detect exact duplicates

The simplest method is hashing normalized examples.

```python
import hashlib


def normalize_text(text: str) -> str:
    return " ".join(
        text.lower().strip().split()
    )


def text_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
```

Now detect overlap:

```python
def find_exact_overlap(
    train_texts,
    validation_texts
):

    train_hashes = {
        text_hash(text)
        for text in train_texts
    }

    overlaps = []

    for text in validation_texts:

        if text_hash(text) in train_hashes:

            overlaps.append(text)

    return overlaps
```

Usage:

```python
train_data = [
    "What is LoRA?",
    "Explain RAG",
    "What is FastAPI?"
]

validation_data = [
    "What is LoRA?",
    "Explain PPO"
]


overlaps = find_exact_overlap(
    train_data,
    validation_data
)

print(overlaps)
```

Output:

```text
['What is LoRA?']
```

---

# 3. Detect overlap between input and output fields

For instruction fine-tuning, you should also check:

```text
Train instruction
Train input
Train output

Validation instruction
Validation input
Validation output
```

For example:

```python
def build_example_text(example):

    return (
        example["instruction"]
        + "\n"
        + example.get("input", "")
        + "\n"
        + example.get("output", "")
    )
```

But be careful: sometimes you want to detect overlap separately.

Example:

```python
def get_question(example):

    return (
        example["instruction"]
        + "\n"
        + example.get("input", "")
    )


def get_answer(example):

    return example.get(
        "output",
        ""
    )
```

Then compare:

```text
Train question ↔ Validation question
Train answer ↔ Validation answer
Train full example ↔ Validation full example
```

---

# 4. Detect duplicate documents before chunking

This is especially important for RAG and document fine-tuning.

Suppose:

```text
Document A
     ↓
Chunking
     ↓
Chunk 1 → Train
Chunk 2 → Validation
```

This creates leakage.

Bad:

```text
Same document
    │
    ├── Train chunks
    │
    └── Validation chunks
```

Instead:

```text
Split by document
        ↓
Chunk after splitting
```

Example:

```python
from sklearn.model_selection import train_test_split


documents = [
    {
        "document_id": "doc_1",
        "text": "..."
    },
    {
        "document_id": "doc_2",
        "text": "..."
    }
]


train_docs, val_docs = train_test_split(
    documents,
    test_size=0.2,
    random_state=42
)


# Chunk only AFTER splitting

train_chunks = chunk_documents(
    train_docs
)

val_chunks = chunk_documents(
    val_docs
)
```

This is a very important production practice.

---

# 5. Detect near duplicates using embeddings

Exact hashes won't detect:

```text
What is LoRA?
```

and:

```text
Explain the purpose of Low Rank Adaptation.
```

Use embeddings and cosine similarity.

Conceptually:

```text
Train example
      ↓
   Embedding
      │
      ├──────────────┐
      │              │
      ▼              ▼
Validation embedding
      │
      ▼
Cosine similarity
      │
      ▼
High similarity?
      │
      ▼
Possible leakage
```

Example:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


train_embeddings = model.encode(
    train_data
)

validation_embeddings = model.encode(
    validation_data
)


similarity_matrix = cosine_similarity(
    validation_embeddings,
    train_embeddings
)
```

Find suspicious matches:

```python
threshold = 0.95


for val_index, row in enumerate(
    similarity_matrix
):

    max_index = row.argmax()

    max_similarity = row[max_index]

    if max_similarity >= threshold:

        print(
            "Possible leakage"
        )

        print(
            "Validation:",
            validation_data[val_index]
        )

        print(
            "Training:",
            train_data[max_index]
        )

        print(
            "Similarity:",
            max_similarity
        )
```

The threshold should be tuned for your dataset. `0.95` is only an example.

---

# 6. Avoid O(N × M) comparisons for large datasets

This code:

```text
Validation examples × Training examples
```

can become very expensive.

For production, use a vector index.

Architecture:

```text
Training Dataset
       │
       ▼
   Embeddings
       │
       ▼
   Vector Index
       │
       ▼
Validation Example
       │
       ▼
Top-K Similar Examples
       │
       ▼
Similarity Threshold
       │
       ├── High → Flag
       │
       └── Low → Pass
```

Example using FAISS:

```python
import faiss
import numpy as np


dimension = train_embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


# Normalize for cosine similarity

train_vectors = (
    train_embeddings
    / np.linalg.norm(
        train_embeddings,
        axis=1,
        keepdims=True
    )
)


validation_vectors = (
    validation_embeddings
    / np.linalg.norm(
        validation_embeddings,
        axis=1,
        keepdims=True
    )
)


index.add(
    train_vectors.astype(
        "float32"
    )
)


scores, indices = index.search(
    validation_vectors.astype(
        "float32"
    ),
    k=5
)
```

Now inspect:

```python
for i in range(
    len(validation_data)
):

    best_score = scores[i][0]

    best_index = indices[i][0]

    if best_score > 0.95:

        print(
            "Potential leakage"
        )

        print(
            "Validation:",
            validation_data[i]
        )

        print(
            "Similar training example:",
            train_data[best_index]
        )

        print(
            "Score:",
            best_score
        )
```

---

# 7. Use fuzzy matching

Embeddings are useful, but character-level similarity can detect lightly modified duplicates.

Example:

```python
from rapidfuzz import fuzz


def find_fuzzy_duplicates(
    train_texts,
    validation_texts,
    threshold=95
):

    matches = []

    for validation_text in validation_texts:

        for train_text in train_texts:

            score = fuzz.token_set_ratio(
                validation_text,
                train_text
            )

            if score >= threshold:

                matches.append({

                    "validation":
                        validation_text,

                    "training":
                        train_text,

                    "score":
                        score
                })

    return matches
```

This can detect:

```text
What is LoRA and how does it work?
```

vs:

```text
How does LoRA work?
```

For large datasets, don't compare every pair directly; use candidate retrieval first.

---

# 8. Detect group leakage

Suppose you have:

```text
customer_id
document_id
conversation_id
user_id
organization_id
```

These groups should often stay together.

Instead of random splitting:

```python
from sklearn.model_selection import train_test_split


train, val = train_test_split(
    dataset,
    test_size=0.2
)
```

Use group splitting:

```python
from sklearn.model_selection import GroupShuffleSplit


splitter = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1,
    random_state=42
)


groups = dataset["customer_id"]


train_indices, val_indices = next(
    splitter.split(
        dataset,
        groups=groups
    )
)
```

Validate:

```python
train_customers = set(
    dataset.iloc[train_indices]["customer_id"]
)

val_customers = set(
    dataset.iloc[val_indices]["customer_id"]
)


leakage = (
    train_customers
    & val_customers
)


assert len(leakage) == 0
```

This ensures:

```text
Customer A
   ↓
Only one split
```

---

# 9. Detect temporal leakage

For time-dependent data:

Bad:

```text
Random split
```

Better:

```text
Past → Train
Future → Validation
```

Example:

```python
import pandas as pd


dataset = pd.DataFrame(
    dataset
)

dataset["timestamp"] = pd.to_datetime(
    dataset["timestamp"]
)

dataset = dataset.sort_values(
    "timestamp"
)


cutoff = pd.Timestamp(
    "2026-01-01"
)


train_data = dataset[
    dataset["timestamp"] < cutoff
]


validation_data = dataset[
    dataset["timestamp"] >= cutoff
]
```

Validate:

```python
assert (
    train_data["timestamp"].max()
    <
    validation_data["timestamp"].min()
)
```

This prevents:

```text
Future
  ↓
Training
  ↓
Predict Past
```

---

# 10. Check feature leakage

For traditional ML, ask:

> Could this feature exist at prediction time?

Example:

```text
Predict:
Will customer default?

Feature:
Loan status after 90 days
```

That is leakage.

You can create a simple feature review:

```python
features = [
    "income",
    "credit_score",
    "loan_status_after_90_days"
]


for feature in features:

    print(
        "Review feature:",
        feature
    )
```

In real production, every feature should have metadata:

```text
Feature name
Source
Creation timestamp
Availability timestamp
Owner
```

Then verify:

```text
feature_available_time
    <=
prediction_time
```

---

# 11. Detect target leakage

This happens when the answer appears directly or indirectly in the input.

Example:

```python
example = {

    "input":
        """
        Customer will churn: YES

        Customer behavior:
        ...
        """,

    "label":
        "YES"
}
```

The answer is already in the input.

For LLM datasets, use rule-based checks:

```python
def target_in_input(
    input_text,
    target
):

    return (
        target.lower()
        in
        input_text.lower()
    )
```

Example:

```python
if target_in_input(
    example["input"],
    example["output"]
):

    print(
        "Possible target leakage"
    )
```

This simple method has false positives, so it should be combined with more task-specific rules.

---

# 12. Prevent preprocessing leakage

This is a subtle but important issue.

Bad:

```text
Full dataset
      ↓
Fit tokenizer/vectorizer/statistics
      ↓
Split
```

Better:

```text
Split first
     │
     ├── Train → Fit preprocessing
     │
     └── Validation → Transform only
```

Traditional ML example:

Bad:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)
```

Then splitting means validation statistics influenced scaling.

Correct:

```python
X_train, X_val = train_test_split(
    X
)


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_val = scaler.transform(
    X_val
)
```

For LLM fine-tuning, the equivalent issue can occur with:

```text
Vocabulary/data-driven preprocessing
Filtering thresholds
Deduplication decisions
Data augmentation
Prompt construction rules
Retrieval corpus construction
```

---

# 13. Detect benchmark contamination

This is extremely important for LLMs.

Suppose you evaluate on a benchmark:

```text
Benchmark questions
```

but those questions were present in:

```text
Pretraining data
Fine-tuning data
Synthetic dataset
```

Then the benchmark score may be inflated.

Check:

```text
Evaluation benchmark
       ↓
Exact match search
       +
Near duplicate search
       +
Semantic search
       ↓
Training corpus
```

Example:

```python
def check_benchmark_contamination(
    benchmark_questions,
    train_questions
):

    benchmark_embeddings = model.encode(
        benchmark_questions
    )

    train_embeddings = model.encode(
        train_questions
    )

    similarities = cosine_similarity(
        benchmark_embeddings,
        train_embeddings
    )

    contamination = []

    for i, row in enumerate(
        similarities
    ):

        max_score = max(row)

        if max_score > 0.95:

            contamination.append({

                "benchmark":
                    benchmark_questions[i],

                "similarity":
                    float(max_score)
            })

    return contamination
```

---

# 14. Build an automated leakage checker

A production dataset pipeline could look like this:

```python
class DataLeakageChecker:

    def check_exact_duplicates(
        self,
        train,
        validation
    ):

        train_hashes = {
            text_hash(x)
            for x in train
        }

        return [

            x for x in validation

            if text_hash(x)
            in train_hashes
        ]

    def check_group_overlap(
        self,
        train_groups,
        validation_groups
    ):

        return (
            set(train_groups)
            &
            set(validation_groups)
        )

    def check_temporal_overlap(
        self,
        train_dates,
        validation_dates
    ):

        return (
            max(train_dates)
            >=
            min(validation_dates)
        )

    def validate(
        self,
        train,
        validation,
        train_groups=None,
        validation_groups=None
    ):

        result = {

            "exact_duplicates":
                self.check_exact_duplicates(
                    train,
                    validation
                ),

            "group_overlap":
                []
        }

        if (
            train_groups
            and
            validation_groups
        ):

            result[
                "group_overlap"
            ] = list(
                self.check_group_overlap(
                    train_groups,
                    validation_groups
                )
            )

        return result
```

Then block training if leakage is detected:

```python
result = checker.validate(
    train_data,
    validation_data
)


if result["exact_duplicates"]:

    raise ValueError(
        "Data leakage detected"
    )
```

In production, I'd usually allow a formal review/quarantine workflow rather than simply crashing the entire pipeline.

---

# 15. A practical leakage report

Instead of returning only `True` or `False`, generate metrics:

```python
report = {

    "train_records": 100000,

    "validation_records": 10000,

    "exact_duplicates": 12,

    "near_duplicates": 87,

    "group_overlap": 0,

    "temporal_leakage": False,

    "benchmark_contamination": 3,

    "status": "FAILED"
}
```

Define thresholds:

```text
Exact duplicates:
0 allowed

Group overlap:
0 allowed

Temporal leakage:
0 allowed

Near duplicates:
Review required

Benchmark contamination:
0 or explicitly documented
```

The exact policy depends on the task.

---

# 16. Production pipeline

A robust pipeline would be:

```text
Raw Dataset
     │
     ▼
Schema Validation
     │
     ▼
Exact Deduplication
     │
     ▼
Near-Duplicate Detection
     │
     ▼
Group Leakage Check
     │
     ▼
Temporal Leakage Check
     │
     ▼
Target Leakage Check
     │
     ▼
Benchmark Contamination Check
     │
     ▼
Leakage Report
     │
     ├──── FAIL ────→ Quarantine
     │
     └──── PASS ────→ Version Dataset
                              │
                              ▼
                          Training
```

---

# 17. The most important checks for LLM fine-tuning

If I were fine-tuning an LLM, my priority order would be:

### 1. Exact train/evaluation overlap

```text
Hash comparison
```

### 2. Near-duplicate overlap

```text
Embeddings + vector search
```

### 3. Document-level leakage

```text
Split documents before chunking
```

### 4. Conversation/user/customer leakage

```text
Group-based splitting
```

### 5. Answer leakage

```text
Check whether labels appear in prompts
```

### 6. Benchmark contamination

```text
Compare evaluation benchmark against training corpus
```

### 7. Temporal leakage

```text
Past → Train
Future → Evaluation
```

---

# Strong interview answer

> **I would detect data leakage before and after dataset splitting. First, I would check exact overlap using normalized hashes. Then I would detect near-duplicates using embeddings and similarity search, especially for paraphrased LLM examples. I would make sure splitting occurs at the correct unit—for example, by document, customer, conversation, or organization rather than randomly splitting chunks or individual rows. For time-dependent data, I would use temporal splits to prevent future information from entering training.**
>
> **I would also inspect prompts for target leakage, verify that preprocessing is fitted only on training data, and check evaluation benchmarks for contamination against the training corpus. Finally, I would automate these checks in the dataset pipeline and generate a leakage report. Datasets exceeding defined thresholds would be quarantined and not promoted to training.**

## Key principle

```text
A model should only learn from information
that would legitimately be available
at prediction time.
```

For your LLM fine-tuning interview preparation, the most important distinction to remember is:

```text
Duplicate leakage
        ↓
Same example appears in train and evaluation

Group leakage
        ↓
Same customer/document appears in both

Temporal leakage
        ↓
Future information enters training

Target leakage
        ↓
Answer is already present in input

Benchmark contamination
        ↓
Evaluation data was seen during training
```
