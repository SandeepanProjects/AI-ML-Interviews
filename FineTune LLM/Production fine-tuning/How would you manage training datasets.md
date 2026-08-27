# How would you manage training datasets in production?

In production, I would not treat a training dataset as just:

```text
data/train.jsonl
```

A dataset is a **versioned, validated, governed, reproducible artifact**.

A good production workflow is:

```text
Raw Data
   │
   ▼
Ingestion
   │
   ▼
Validation
   │
   ▼
PII / Sensitive Data Handling
   │
   ▼
Cleaning & Deduplication
   │
   ▼
Quality Filtering
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Dataset Versioning
   │
   ▼
Immutable Dataset Artifact
   │
   ▼
Training
```

---

# 1. What does dataset management mean?

Dataset management means controlling:

* where training data comes from
* who created it
* when it was created
* which transformations were applied
* dataset versions
* train/validation/test splits
* data quality
* duplicates
* PII and sensitive information
* access control
* reproducibility
* lineage

For example:

```text
Dataset: customer_support
Version: v1.0
Records: 100,000
Created: 2026-08-27

Source:
- Customer support conversations

Transformations:
- PII masking
- Deduplication
- Quality filtering

Split:
- Train: 80,000
- Validation: 10,000
- Test: 10,000
```

If somebody asks:

> Which dataset trained model version 17?

You should be able to answer exactly.

---

# 2. Production dataset architecture

A useful directory structure:

```text
datasets/
│
├── raw/
│   ├── support_data_2026_01.jsonl
│   └── support_data_2026_02.jsonl
│
├── processed/
│   ├── customer_support_v1/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   ├── test.jsonl
│   │   └── metadata.json
│   │
│   └── customer_support_v2/
│
├── schemas/
│   └── training_schema.py
│
└── manifests/
    └── dataset_manifest.json
```

In cloud production systems, these would usually live in object storage rather than a local disk.

For example:

```text
Object Storage
│
├── raw/
├── processed/
├── datasets/
│   ├── customer-support/
│   │   ├── v1
│   │   ├── v2
│   │   └── v3
│
└── model-artifacts/
```

---

# 3. Define a dataset schema

Never trust raw data.

Suppose we are training a customer-support LLM.

```python
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    id: str

    instruction: str = Field(
        min_length=1
    )

    input: str | None = None

    output: str = Field(
        min_length=1
    )

    source: str

    conversation_id: str | None = None

    dataset_version: str | None = None
```

Example:

```python
example = {
    "id": "msg_001",
    "instruction": "Answer the customer politely.",
    "input": "I forgot my password.",
    "output": (
        "You can reset your password by "
        "selecting Forgot Password."
    ),
    "source": "customer_support",
    "conversation_id": "conv_1001"
}
```

Validation:

```python
validated = TrainingExample.model_validate(
    example
)

print(validated)
```

This catches malformed records early.

---

# 4. Build a data ingestion layer

Don't mix raw data loading with training.

```python
# app/data/loader.py

import json


def load_jsonl(
    file_path: str
) -> list[dict]:

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            records.append(record)

    return records
```

Usage:

```python
records = load_jsonl(
    "datasets/raw/support.jsonl"
)

print(
    f"Loaded {len(records)} records"
)
```

---

# 5. Validate all records

```python
from pydantic import ValidationError


def validate_records(
    records: list[dict]
):

    valid_records = []

    invalid_records = []

    for record in records:

        try:

            validated = (
                TrainingExample
                .model_validate(record)
            )

            valid_records.append(
                validated.model_dump()
            )

        except ValidationError as error:

            invalid_records.append({
                "record": record,
                "error": str(error)
            })

    return (
        valid_records,
        invalid_records
    )
```

Usage:

```python
valid, invalid = validate_records(
    records
)

print(
    f"Valid: {len(valid)}"
)

print(
    f"Invalid: {len(invalid)}"
)
```

In production, invalid records should be logged with a reason.

Example:

```text
Total Records:      1,000,000
Valid:                982,000
Invalid:               18,000

Reasons:

Missing output:         10,000
Invalid JSON:            5,000
Output too short:        3,000
```

---

# 6. Remove PII

This is particularly important if you train using production conversations.

Example:

```text
Customer:
My email is john@example.com
```

You may not want the model to memorize:

```text
john@example.com
```

A simplified implementation:

```python
import re


EMAIL_PATTERN = re.compile(
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
)

PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b"
)


def mask_pii(
    text: str
) -> str:

    text = EMAIL_PATTERN.sub(
        "[EMAIL]",
        text
    )

    text = PHONE_PATTERN.sub(
        "[PHONE]",
        text
    )

    return text
```

Apply it:

```python
def sanitize_record(
    record: dict
) -> dict:

    record = record.copy()

    for field in [
        "instruction",
        "input",
        "output"
    ]:

        if record.get(field):

            record[field] = mask_pii(
                record[field]
            )

    return record
```

Example:

```python
record = {
    "instruction": "Answer politely",
    "input": "My email is john@example.com",
    "output": "We will contact you."
}

cleaned = sanitize_record(
    record
)

print(cleaned)
```

Output:

```text
{
    'instruction': 'Answer politely',
    'input': 'My email is [EMAIL]',
    'output': 'We will contact you.'
}
```

For a real enterprise system, I would use stronger PII detection and policy-based redaction rather than only regex.

---

# 7. Deduplicate the dataset

Duplicate examples cause problems.

Example:

```text
Example A:
How do I reset my password?

Example B:
How do I reset my password?
```

Exact deduplication:

```python
import hashlib
import json


def create_record_hash(
    record: dict
) -> str:

    normalized = json.dumps(
        {
            "instruction":
                record.get(
                    "instruction",
                    ""
                ).strip().lower(),

            "input":
                record.get(
                    "input",
                    ""
                ).strip().lower(),

            "output":
                record.get(
                    "output",
                    ""
                ).strip().lower()
        },
        sort_keys=True
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
```

Remove duplicates:

```python
def remove_duplicates(
    records: list[dict]
):

    seen = set()

    unique_records = []

    for record in records:

        record_hash = (
            create_record_hash(record)
        )

        if record_hash in seen:
            continue

        seen.add(record_hash)

        unique_records.append(record)

    return unique_records
```

For large datasets, exact matching is not enough.

You also want to detect near duplicates:

```text
How can I reset my password?

How do I reset my password?

How can I change my password?
```

Production approaches include:

* MinHash
* SimHash
* locality-sensitive hashing
* embedding similarity

---

# 8. Quality filtering

Schema-valid data can still be poor training data.

Example:

```text
Question:
How do I reset my password?

Answer:
asdfgh
```

Create quality rules:

```python
def passes_quality_checks(
    record: dict
) -> bool:

    output = record.get(
        "output",
        ""
    ).strip()

    instruction = record.get(
        "instruction",
        ""
    ).strip()

    if len(instruction) < 5:
        return False

    if len(output) < 10:
        return False

    if len(output) > 5000:
        return False

    return True
```

Filter:

```python
def filter_quality(
    records: list[dict]
):

    return [

        record

        for record in records

        if passes_quality_checks(
            record
        )
    ]
```

More advanced production checks:

```text
Schema Validation
        +
Length Validation
        +
Language Detection
        +
Toxicity Detection
        +
PII Detection
        +
Duplicate Detection
        +
LLM Quality Judge
```

---

# 9. Avoid data leakage during splitting

This is extremely important.

Bad:

```text
Conversation 100:

Message 1 → Train

Message 2 → Test
```

The model has effectively already seen the conversation.

Instead:

```text
Conversation 100 → Train

Conversation 101 → Validation

Conversation 102 → Test
```

Use group splitting.

```python
from sklearn.model_selection import (
    GroupShuffleSplit
)


def group_split(
    records
):

    groups = [

        record["conversation_id"]

        for record in records
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_indices, temp_indices = next(

        splitter.split(
            records,
            groups=groups
        )
    )

    train = [
        records[i]
        for i in train_indices
    ]

    temp = [
        records[i]
        for i in temp_indices
    ]

    return train, temp
```

Then split the temporary dataset again:

```text
80% → Train

10% → Validation

10% → Test
```

---

# 10. Detect train/test overlap

Even after splitting, similar examples may exist.

Example:

### Train

```text
How do I reset my password?
```

### Test

```text
How can I reset my password?
```

You can use embeddings to identify suspicious overlap.

Conceptually:

```text
Train Embeddings
      │
      ▼
Vector Index
      │
      │ Compare
      ▼
Test Embedding
      │
      ▼
Similarity Score
```

Simplified code:

```python
import numpy as np


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )
```

Then:

```python
def detect_overlap(
    train_embeddings,
    test_embedding,
    threshold=0.95
):

    similarities = [

        cosine_similarity(
            train_embedding,
            test_embedding
        )

        for train_embedding
        in train_embeddings
    ]

    max_similarity = max(
        similarities
    )

    return max_similarity > threshold
```

For millions of examples, you would use a vector index rather than comparing every example.

---

# 11. Version the dataset

This is one of the most important parts.

Suppose:

```text
Dataset v1
↓
Train Model v1

Dataset v2
↓
Train Model v2
```

You should never overwrite v1.

Instead:

```text
customer-support/
│
├── v1/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── metadata.json
│
├── v2/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── metadata.json
│
└── v3/
```

Dataset metadata:

```json
{
    "dataset_name": "customer_support",
    "version": "v3",
    "created_at": "2026-08-27",
    "record_count": 850000,
    "train_count": 680000,
    "validation_count": 85000,
    "test_count": 85000,
    "source": "support_conversations",
    "transformations": [
        "pii_masking",
        "deduplication",
        "quality_filter"
    ]
}
```

---

# 12. Create a dataset manifest

A manifest describes exactly what was used.

```python
from dataclasses import (
    dataclass,
    asdict
)

from datetime import datetime
import json


@dataclass
class DatasetManifest:

    dataset_name: str

    version: str

    created_at: str

    record_count: int

    train_count: int

    validation_count: int

    test_count: int

    source: str

    transformations: list[str]
```

Create:

```python
manifest = DatasetManifest(

    dataset_name=
        "customer_support",

    version="v3",

    created_at=
        datetime.utcnow().isoformat(),

    record_count=100000,

    train_count=80000,

    validation_count=10000,

    test_count=10000,

    source=
        "support_conversations",

    transformations=[
        "pii_masking",
        "deduplication",
        "quality_filter"
    ]
)
```

Save:

```python
def save_manifest(
    manifest,
    file_path
):

    with open(
        file_path,
        "w"
    ) as file:

        json.dump(
            asdict(manifest),
            file,
            indent=2
        )
```

---

# 13. Dataset fingerprinting

Version numbers alone are not enough.

You want to know:

> Did the actual contents change?

Create a dataset fingerprint.

```python
import hashlib


def dataset_fingerprint(
    records
) -> str:

    hasher = hashlib.sha256()

    for record in sorted(
        records,
        key=lambda x: x["id"]
    ):

        text = json.dumps(
            record,
            sort_keys=True
        )

        hasher.update(
            text.encode()
        )

    return hasher.hexdigest()
```

Usage:

```python
fingerprint = dataset_fingerprint(
    train_records
)

print(fingerprint)
```

Example:

```text
Dataset: customer-support-v3

Fingerprint:
a84bcf7e1a29d8...
```

This improves reproducibility.

---

# 14. Build a dataset pipeline

Now combine everything.

```python
def prepare_dataset(
    raw_records
):

    # 1. Validate
    valid_records, invalid_records = (
        validate_records(
            raw_records
        )
    )

    # 2. Sanitize PII
    sanitized = [

        sanitize_record(record)

        for record
        in valid_records
    ]

    # 3. Remove duplicates
    unique_records = (
        remove_duplicates(
            sanitized
        )
    )

    # 4. Quality filtering
    quality_records = (
        filter_quality(
            unique_records
        )
    )

    return {

        "records":
            quality_records,

        "invalid_count":
            len(invalid_records),

        "final_count":
            len(quality_records)
    }
```

Usage:

```python
raw_records = load_jsonl(
    "datasets/raw/support.jsonl"
)

result = prepare_dataset(
    raw_records
)

print(
    result["final_count"]
)
```

---

# 15. Create train/validation/test datasets

```python
from sklearn.model_selection import (
    train_test_split
)


def split_dataset(
    records
):

    train, remaining = train_test_split(

        records,

        test_size=0.2,

        random_state=42
    )

    validation, test = train_test_split(

        remaining,

        test_size=0.5,

        random_state=42
    )

    return {
        "train": train,
        "validation": validation,
        "test": test
    }
```

Result:

```text
Total: 100,000

Train:       80,000
Validation:  10,000
Test:        10,000
```

For conversational data, use a group-based split as explained earlier.

---

# 16. Save immutable dataset versions

```python
import os


def save_jsonl(
    records,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(record)
                + "\n"
            )
```

Save:

```python
save_jsonl(
    splits["train"],
    "datasets/customer_support/v3/train.jsonl"
)

save_jsonl(
    splits["validation"],
    "datasets/customer_support/v3/validation.jsonl"
)

save_jsonl(
    splits["test"],
    "datasets/customer_support/v3/test.jsonl"
)
```

Important principle:

```text
v3 should be immutable
```

If data changes:

```text
Create v4
```

Do not silently modify:

```text
v3/train.jsonl
```

Otherwise you lose reproducibility.

---

# 17. Link model versions to dataset versions

This is critical in production.

```text
Model v10
    │
    ├── Base Model: LLM-X
    │
    ├── Dataset: customer-support-v3
    │
    ├── Dataset Fingerprint
    │
    ├── Training Config
    │
    └── Evaluation Results
```

Example:

```python
model_metadata = {

    "model_version":
        "support-model-v10",

    "base_model":
        "base-llm-v1",

    "dataset_name":
        "customer-support",

    "dataset_version":
        "v3",

    "dataset_fingerprint":
        fingerprint,

    "training_config": {

        "learning_rate": 2e-4,

        "epochs": 3,

        "lora_rank": 16
    }
}
```

Now you can reproduce training.

---

# 18. Use data lineage

Production systems should answer:

```text
Where did this model come from?
```

Example:

```text
Raw Data
  │
  ├── Support Database
  ├── Knowledge Base
  └── Human Annotation
           │
           ▼
     Dataset Pipeline
           │
           ▼
     customer-support-v3
           │
           ▼
       Model v10
```

This is called **data lineage**.

A simplified lineage record:

```python
lineage = {

    "raw_sources": [
        "support_db_2026_06",
        "support_db_2026_07"
    ],

    "dataset_version":
        "customer-support-v3",

    "transformations": [
        "PII masking",
        "deduplication",
        "quality filtering"
    ],

    "trained_model":
        "support-model-v10"
}
```

---

# 19. Add automated dataset tests

Treat datasets like software.

Example tests:

```python
def test_no_empty_outputs(
    dataset
):

    for record in dataset:

        assert record[
            "output"
        ].strip()
```

Test duplicate IDs:

```python
def test_unique_ids(
    dataset
):

    ids = [

        record["id"]

        for record
        in dataset
    ]

    assert len(ids) == len(
        set(ids)
    )
```

Test leakage:

```python
def test_no_split_overlap(
    train,
    test
):

    train_ids = {
        item["id"]
        for item in train
    }

    test_ids = {
        item["id"]
        for item in test
    }

    overlap = (
        train_ids
        &
        test_ids
    )

    assert not overlap
```

Test output quality:

```python
def test_output_length(
    dataset
):

    for item in dataset:

        assert len(
            item["output"]
        ) >= 10
```

This is similar to:

```text
Unit Testing
        ↓
Data Testing
```

---

# 20. Dataset CI/CD pipeline

You can automatically validate datasets.

```text
New Dataset
     │
     ▼
Schema Tests
     │
     ▼
PII Tests
     │
     ▼
Duplicate Tests
     │
     ▼
Leakage Tests
     │
     ▼
Quality Tests
     │
     ▼
Dataset Version
     │
     ├── FAIL → Reject Dataset
     │
     └── PASS
             │
             ▼
       Ready for Training
```

Example:

```python
def validate_dataset_pipeline(
    dataset
):

    validate_schema(dataset)

    validate_no_duplicates(dataset)

    validate_no_empty_outputs(dataset)

    validate_no_pii(dataset)

    validate_quality(dataset)

    return True
```

Then:

```python
if validate_dataset_pipeline(
    dataset
):

    print(
        "Dataset approved"
    )
```

---

# 21. Dataset registry

For multiple datasets, maintain a registry.

```python
DATASET_REGISTRY = {

    "customer_support_v1": {
        "path":
            "datasets/customer_support/v1",

        "status":
            "archived"
    },

    "customer_support_v2": {
        "path":
            "datasets/customer_support/v2",

        "status":
            "production"
    }
}
```

A more production-oriented design:

```text
Dataset Registry

Dataset Name
     │
     ├── Version
     │
     ├── Owner
     │
     ├── Created At
     │
     ├── Schema Version
     │
     ├── Fingerprint
     │
     ├── Record Count
     │
     ├── Quality Score
     │
     └── Approval Status
```

---

# 22. What tools can you use?

A production stack might look like:

```text
Data Storage
    │
    ├── Object Storage
    │
Dataset Versioning
    │
    ├── DVC
    ├── LakeFS
    └── Delta Lake
    │
Data Validation
    │
    ├── Great Expectations
    ├── Pandera
    └── Pydantic
    │
Experiment Tracking
    │
    ├── MLflow
    └── Weights & Biases
    │
Dataset / Model Registry
    │
    ├── MLflow
    └── Custom Metadata Database
```

The exact stack depends on company infrastructure.

---

# 23. Complete production pipeline

Here is a simplified orchestrator:

```python
def run_dataset_pipeline(
    raw_file: str,
    dataset_version: str
):

    print("Loading raw data...")

    raw_records = load_jsonl(
        raw_file
    )


    print("Validating data...")

    valid_records, invalid_records = (
        validate_records(
            raw_records
        )
    )


    print("Removing PII...")

    sanitized_records = [

        sanitize_record(record)

        for record
        in valid_records
    ]


    print("Deduplicating...")

    unique_records = (
        remove_duplicates(
            sanitized_records
        )
    )


    print("Filtering quality...")

    clean_records = (
        filter_quality(
            unique_records
        )
    )


    print("Splitting dataset...")

    splits = split_dataset(
        clean_records
    )


    print("Creating fingerprint...")

    fingerprint = (
        dataset_fingerprint(
            clean_records
        )
    )


    print("Saving dataset...")

    base_path = (
        f"datasets/customer_support/"
        f"{dataset_version}"
    )


    save_jsonl(
        splits["train"],
        f"{base_path}/train.jsonl"
    )

    save_jsonl(
        splits["validation"],
        f"{base_path}/validation.jsonl"
    )

    save_jsonl(
        splits["test"],
        f"{base_path}/test.jsonl"
    )


    return {

        "dataset_version":
            dataset_version,

        "fingerprint":
            fingerprint,

        "raw_count":
            len(raw_records),

        "final_count":
            len(clean_records),

        "invalid_count":
            len(invalid_records),

        "train_count":
            len(splits["train"]),

        "validation_count":
            len(
                splits["validation"]
            ),

        "test_count":
            len(
                splits["test"]
            )
    }
```

Usage:

```python
result = run_dataset_pipeline(
    raw_file=
        "datasets/raw/support.jsonl",

    dataset_version="v3"
)

print(result)
```

---

# 24. How I would manage 1 million examples

For a dataset containing:

```text
1,000,000 conversations
```

I would not process everything in Python lists on one machine.

I would use:

```text
                Data Sources
                     │
                     ▼
                Object Storage
                     │
                     ▼
            Distributed Processing
               Spark / Ray
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        PII       Dedup      Quality
          │          │          │
          └──────────┼──────────┘
                     ▼
              Dataset Version
                     │
                     ▼
               Object Storage
                     │
                     ▼
              Training Cluster
```

Important ideas:

* process data in batches
* avoid loading all data into memory
* partition datasets
* use distributed processing
* cache intermediate artifacts
* make transformations reproducible
* maintain immutable versions

---

# 25. Interview-ready answer

> **I manage training datasets as versioned and governed artifacts rather than files. The pipeline starts with raw-data ingestion, followed by schema validation, PII handling, deduplication, quality filtering, and leakage detection. I then create train, validation, and held-out test splits, taking care to split by entity or conversation when necessary to avoid correlated examples across splits.**
>
> **Each processed dataset is immutable and receives a version, metadata manifest, and fingerprint. The manifest records the source, transformations, schema version, record counts, and split information. Every trained model stores a reference to the exact dataset version and fingerprint, which provides reproducibility and lineage.**
>
> **I also add automated data tests to CI/CD for schema failures, duplicates, empty outputs, PII leakage, and split overlap. For large datasets, I use object storage and distributed processing rather than loading everything into memory.**

## Key takeaway

```text
Production Dataset
      ≠
CSV file
```

It is:

```text
Versioned
    +
Validated
    +
Cleaned
    +
Secure
    +
Immutable
    +
Tested
    +
Reproducible
    +
Traceable
```

That is how I would manage datasets for a **production LLM fine-tuning pipeline**.
