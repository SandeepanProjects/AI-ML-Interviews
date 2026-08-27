# How would you version datasets in production?

Dataset versioning means you can answer:

> **Exactly which data was used to train this model, what changed from the previous dataset, and can I reproduce it?**

For example:

```text
Model: support-model-v3
        │
        ▼
Dataset: customer-support-v5
        │
        ├── Raw data sources
        ├── Data transformation version
        ├── Dataset fingerprint
        ├── Schema version
        └── Train/validation/test split
```

---

# 1. Why version datasets?

Imagine you trained:

```text
support-model-v1 → dataset-v1
support-model-v2 → dataset-v2
support-model-v3 → dataset-v3
```

Now model v3 performs badly.

You need to answer:

```text
What changed?
```

Without dataset versioning:

```text
train.jsonl  ❌
train_final.jsonl ❌
train_latest.jsonl ❌
train_final_final.jsonl ❌
```

You cannot reliably know.

Instead:

```text
customer-support/
├── v1/
├── v2/
├── v3/
├── v4/
└── v5/
```

Each version is immutable.

---

# 2. Basic dataset version structure

```text
datasets/
│
└── customer-support/
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
        ├── train.jsonl
        ├── validation.jsonl
        ├── test.jsonl
        └── metadata.json
```

The key rule:

> **Never modify v1 after it has been used for training.**

If the dataset changes:

```text
v1 → create v2
```

Not:

```text
v1 → overwrite train.jsonl ❌
```

---

# 3. What should be stored in a dataset version?

Every dataset version should have metadata.

```json
{
  "dataset_name": "customer-support",
  "version": "v5",
  "created_at": "2026-08-27",

  "schema_version": "v2",

  "source": [
    "support_database",
    "human_annotations"
  ],

  "record_count": 1000000,

  "splits": {
    "train": 800000,
    "validation": 100000,
    "test": 100000
  },

  "transformations": [
    "pii_masking",
    "deduplication",
    "quality_filtering"
  ],

  "fingerprint": "a84bcf..."
}
```

This is the **dataset manifest**.

---

# 4. Dataset version schema using Pydantic

```python
from pydantic import BaseModel
from datetime import datetime


class DatasetVersion(BaseModel):

    dataset_name: str

    version: str

    created_at: datetime

    schema_version: str

    sources: list[str]

    record_count: int

    train_count: int

    validation_count: int

    test_count: int

    transformations: list[str]

    fingerprint: str
```

Create a version:

```python
from datetime import datetime


dataset_metadata = DatasetVersion(

    dataset_name="customer-support",

    version="v5",

    created_at=datetime.utcnow(),

    schema_version="v2",

    sources=[
        "support_database",
        "human_annotations"
    ],

    record_count=1_000_000,

    train_count=800_000,

    validation_count=100_000,

    test_count=100_000,

    transformations=[
        "pii_masking",
        "deduplication",
        "quality_filtering"
    ],

    fingerprint="abc123"
)
```

---

# 5. Dataset fingerprinting

A version number is not enough.

Consider:

```text
Dataset v5
```

Someone accidentally changes one record.

The directory still says:

```text
v5
```

But the contents changed.

So create a **fingerprint/hash**.

## File-level hashing

```python
import hashlib


def calculate_file_hash(
    file_path: str
) -> str:

    hasher = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while chunk := file.read(8192):

            hasher.update(chunk)

    return hasher.hexdigest()
```

Usage:

```python
hash_value = calculate_file_hash(
    "train.jsonl"
)

print(hash_value)
```

Output:

```text
a84bcf72f0f9...
```

If even one byte changes:

```text
Old Hash:
a84bcf72

New Hash:
9d3e7a10
```

You know the dataset changed.

---

# 6. Create a fingerprint for multiple files

Your dataset usually has:

```text
train.jsonl
validation.jsonl
test.jsonl
```

Create a fingerprint for all of them.

```python
import hashlib


def calculate_dataset_fingerprint(
    file_paths: list[str]
) -> str:

    dataset_hasher = hashlib.sha256()

    for path in sorted(file_paths):

        file_hash = calculate_file_hash(
            path
        )

        dataset_hasher.update(
            file_hash.encode()
        )

    return (
        dataset_hasher.hexdigest()
    )
```

Usage:

```python
fingerprint = (
    calculate_dataset_fingerprint(
        [
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl"
        ]
    )
)
```

Now:

```text
Dataset v5
        │
        ▼
Fingerprint
a84bcf72...
```

---

# 7. Version based on data changes

You can use semantic-style versions.

```text
v1.0.0
```

### Major change

```text
v1.0.0 → v2.0.0
```

Examples:

* completely new dataset
* changed schema
* changed task
* changed label format

Example:

```text
v1:
{
  "question": "...",
  "answer": "..."
}
```

```text
v2:
{
  "messages": [...]
}
```

---

### Minor change

```text
v1.0.0 → v1.1.0
```

Examples:

* add 100,000 new examples
* add new domain
* add additional languages

---

### Patch change

```text
v1.1.0 → v1.1.1
```

Examples:

* remove bad examples
* fix labeling errors
* correct metadata

The exact rules can vary by organization.

---

# 8. Version raw data separately from processed data

This is important.

Don't do:

```text
Raw Data → Processing → Dataset
```

without versioning the source.

Instead:

```text
Raw Dataset v3
       │
       ▼
Transformation Pipeline v7
       │
       ▼
Processed Dataset v12
```

Example:

```text
datasets/
│
├── raw/
│   ├── support/
│   │   ├── v1/
│   │   └── v2/
│
└── processed/
    ├── support-training/
    │   ├── v1/
    │   └── v2/
```

This gives better lineage.

---

# 9. Version the transformation pipeline

The data transformation code also affects the dataset.

Example:

```python
def clean_text_v1(text):
    return text.strip()
```

Later:

```python
def clean_text_v2(text):

    text = text.strip().lower()

    return text
```

Same raw data.

Different processed dataset.

Therefore, store:

```text
Raw Data Version: raw-v3

Transformation Version: pipeline-v2

Processed Dataset: train-v5
```

You should also store the Git commit:

```python
metadata = {
    "raw_dataset_version": "raw-v3",
    "transformation_version": "pipeline-v2",
    "code_commit": "a4f87c2",
    "processed_dataset_version": "v5"
}
```

---

# 10. Build an immutable dataset version

```python
from pathlib import Path


def create_dataset_version(
    dataset_name: str,
    version: str,
    root: str = "./datasets"
):

    dataset_path = (
        Path(root)
        / dataset_name
        / version
    )

    if dataset_path.exists():

        raise ValueError(
            f"Dataset version already exists: "
            f"{version}"
        )

    dataset_path.mkdir(
        parents=True
    )

    return dataset_path
```

Usage:

```python
path = create_dataset_version(

    dataset_name="customer-support",

    version="v5"
)

print(path)
```

Output:

```text
datasets/customer-support/v5
```

If somebody tries:

```python
create_dataset_version(
    "customer-support",
    "v5"
)
```

Again:

```text
ValueError:
Dataset version already exists
```

This prevents accidental overwrites.

---

# 11. Save dataset and manifest

```python
import json


def save_jsonl(
    records: list[dict],
    output_path: str
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(record)
                + "\n"
            )
```

Save manifest:

```python
def save_manifest(
    metadata: DatasetVersion,
    output_path: str
):

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            metadata.model_dump(
                mode="json"
            ),
            file,
            indent=2
        )
```

Complete usage:

```python
dataset_path = create_dataset_version(
    "customer-support",
    "v5"
)

save_jsonl(
    train_records,
    dataset_path / "train.jsonl"
)

save_jsonl(
    validation_records,
    dataset_path / "validation.jsonl"
)

save_jsonl(
    test_records,
    dataset_path / "test.jsonl"
)

save_manifest(
    dataset_metadata,
    dataset_path / "metadata.json"
)
```

---

# 12. Detect what changed between versions

Suppose:

```text
v4 → v5
```

You should know what changed.

Create record IDs:

```python
def get_record_ids(
    records
):

    return {
        record["id"]
        for record in records
    }
```

Compare:

```python
def compare_datasets(
    old_records,
    new_records
):

    old_ids = get_record_ids(
        old_records
    )

    new_ids = get_record_ids(
        new_records
    )

    added = (
        new_ids - old_ids
    )

    removed = (
        old_ids - new_ids
    )

    return {

        "added": len(added),

        "removed": len(removed),

        "unchanged": len(
            old_ids & new_ids
        )
    }
```

Usage:

```python
changes = compare_datasets(
    dataset_v4,
    dataset_v5
)

print(changes)
```

Output:

```text
{
    "added": 120000,
    "removed": 5000,
    "unchanged": 875000
}
```

---

# 13. Track changed records

IDs alone don't detect modified records.

Create a hash per record.

```python
import hashlib
import json


def record_hash(
    record: dict
):

    normalized = json.dumps(
        record,
        sort_keys=True
    )

    return hashlib.sha256(
        normalized.encode()
    ).hexdigest()
```

Compare:

```python
def find_modified_records(
    old_records,
    new_records
):

    old_map = {

        record["id"]:
        record_hash(record)

        for record
        in old_records
    }

    new_map = {

        record["id"]:
        record_hash(record)

        for record
        in new_records
    }

    modified = []

    common_ids = (
        old_map.keys()
        &
        new_map.keys()
    )

    for record_id in common_ids:

        if (
            old_map[record_id]
            !=
            new_map[record_id]
        ):

            modified.append(
                record_id
            )

    return modified
```

Now you can produce:

```text
Dataset v4 → v5

Added:      120,000
Removed:      5,000
Modified:     2,400
```

---

# 14. Version train/validation/test splits

The split itself is part of the dataset version.

Suppose:

```text
Dataset v5
```

First run:

```text
Record A → Train
```

Second run:

```text
Record A → Test
```

Your evaluation is no longer reproducible.

So preserve split assignments.

Example:

```json
{
  "record_1": "train",
  "record_2": "train",
  "record_3": "validation",
  "record_4": "test"
}
```

Code:

```python
import json


def save_split_assignments(
    splits: dict,
    output_path: str
):

    assignments = {}

    for split_name, records in splits.items():

        for record in records:

            assignments[
                record["id"]
            ] = split_name

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            assignments,
            file,
            indent=2
        )
```

Save:

```text
v5/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── split_assignments.json
└── metadata.json
```

---

# 15. Dataset registry

In production, you can maintain a dataset registry.

For example, PostgreSQL:

```sql
CREATE TABLE dataset_versions (

    id UUID PRIMARY KEY,

    dataset_name VARCHAR(255),

    version VARCHAR(50),

    fingerprint TEXT,

    raw_dataset_version VARCHAR(50),

    transformation_version VARCHAR(50),

    code_commit VARCHAR(255),

    record_count INTEGER,

    status VARCHAR(50),

    created_at TIMESTAMP,

    UNIQUE (
        dataset_name,
        version
    )
);
```

Now you can query:

```sql
SELECT
    dataset_name,
    version,
    fingerprint,
    record_count

FROM dataset_versions

WHERE dataset_name =
    'customer-support';
```

---

# 16. Dataset lifecycle

A production dataset should have stages:

```text
RAW
 │
 ▼
VALIDATING
 │
 ▼
CANDIDATE
 │
 ▼
APPROVED
 │
 ▼
TRAINING
 │
 ▼
ARCHIVED
```

Example:

```python
from enum import Enum


class DatasetStage(str, Enum):

    RAW = "raw"

    CANDIDATE = "candidate"

    APPROVED = "approved"

    TRAINING = "training"

    ARCHIVED = "archived"
```

---

# 17. Dataset quality gates

Before promoting a dataset:

```text
Candidate
    │
    ▼
Quality Gates
```

Check:

```text
✓ Schema validation

✓ No duplicate IDs

✓ No PII

✓ No train/test leakage

✓ Minimum output quality

✓ Correct class distribution

✓ Correct language distribution

✓ No corrupted records
```

Example:

```python
def dataset_quality_gate(
    metrics: dict
):

    if metrics[
        "invalid_records"
    ] > 0:

        return False

    if metrics[
        "duplicate_rate"
    ] > 0.01:

        return False

    if metrics[
        "pii_count"
    ] > 0:

        return False

    if metrics[
        "train_test_overlap"
    ] > 0:

        return False

    return True
```

Usage:

```python
if dataset_quality_gate(
    dataset_metrics
):

    stage = "approved"

else:

    stage = "rejected"
```

---

# 18. Connect dataset version to model version

This is critical.

```text
Dataset v5
      │
      ▼
Model v1.0
```

Your model metadata should include:

```python
model_metadata = {

    "model_version":
        "support-model-v1.0",

    "dataset_name":
        "customer-support",

    "dataset_version":
        "v5",

    "dataset_fingerprint":
        "a84bcf72",

    "training_code_commit":
        "9fa32b"
}
```

Now you can answer:

> Which data trained this model?

```text
customer-support v5
```

And:

> Is this exact dataset still available?

```text
Yes — identified by fingerprint a84bcf72.
```

---

# 19. A complete dataset versioning pipeline

```python
def create_versioned_dataset(
    dataset_name: str,
    version: str,
    records: list[dict],
    metadata: dict
):

    # 1. Create immutable version
    path = create_dataset_version(
        dataset_name,
        version
    )

    # 2. Split dataset
    splits = split_dataset(
        records
    )

    # 3. Save data
    save_jsonl(
        splits["train"],
        path / "train.jsonl"
    )

    save_jsonl(
        splits["validation"],
        path / "validation.jsonl"
    )

    save_jsonl(
        splits["test"],
        path / "test.jsonl"
    )

    # 4. Calculate fingerprint
    fingerprint = (
        calculate_dataset_fingerprint(
            [
                str(
                    path / "train.jsonl"
                ),
                str(
                    path / "validation.jsonl"
                ),
                str(
                    path / "test.jsonl"
                )
            ]
        )
    )

    # 5. Add metadata
    metadata["dataset_name"] = (
        dataset_name
    )

    metadata["version"] = version

    metadata["fingerprint"] = fingerprint

    # 6. Save manifest
    with open(
        path / "metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    return path
```

Usage:

```python
path = create_versioned_dataset(

    dataset_name=
        "customer-support",

    version="v5",

    records=
        cleaned_records,

    metadata={

        "raw_dataset_version":
            "raw-v3",

        "transformation_version":
            "pipeline-v2",

        "code_commit":
            "abc123",

        "record_count":
            len(cleaned_records)
    }
)
```

---

# 20. Production architecture

A real-world architecture could look like:

```text
                  Data Sources
                       │
                       ▼
                  Raw Storage
                       │
                       │ Raw Dataset Version
                       ▼
                Processing Pipeline
                       │
                       │ Git Commit
                       │ Pipeline Version
                       ▼
                 Quality Checks
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Failed              Approved
                                 │
                                 ▼
                       Dataset Registry
                                 │
                                 ▼
                       Dataset v5 Immutable
                                 │
                                 ▼
                         Model Training
                                 │
                                 ▼
                           Model v1.2
```

---

# Tools I would consider

For a real production project, common categories are:

```text
Dataset Versioning:
- DVC
- lakeFS
- Delta Lake

Experiment / Model Tracking:
- MLflow

Data Validation:
- Great Expectations
- Pandera

Storage:
- S3 / GCS / Azure Blob

Metadata:
- PostgreSQL

Large-scale Processing:
- Spark
- Ray
```

The important thing is not the specific tool. The architecture matters:

```text
Immutable Data
+
Version
+
Fingerprint
+
Manifest
+
Lineage
+
Quality Validation
+
Reproducibility
```

---

# Interview-ready answer

> **I version datasets as immutable, reproducible artifacts. Every dataset version has a unique version ID and fingerprint, and I never overwrite a dataset that has been used for training. I maintain a manifest containing the raw data sources, schema version, transformation pipeline version, code commit, record counts, split assignments, and quality metrics.**
>
> **I version raw data separately from processed training data because changes in either the source or transformation logic can produce a different dataset. I also preserve train, validation, and test split assignments to prevent leakage and ensure reproducible evaluation.**
>
> **Each model version is linked to the exact dataset version and fingerprint, which provides complete lineage from raw data through processing, training, evaluation, and deployment. In production, I would typically use object storage for immutable artifacts, a tool such as DVC or lakeFS for data versioning, MLflow for experiment tracking, and a metadata registry for lineage and governance.**

## One-line summary

```text
Dataset Version =
Data + Schema + Transformations + Splits + Fingerprint + Metadata + Lineage
```

That is the core of **production-grade dataset versioning**.
