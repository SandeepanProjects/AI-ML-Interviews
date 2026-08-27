# How would you version models in production?

For an ML/LLM system, **model versioning means being able to answer exactly**:

> Which model is running in production, what data trained it, what code/configuration produced it, what evaluation results it achieved, and how do we roll back?

A production model should **never** just be:

```text
model_final/
model_final_v2/
model_final_latest/
```

Instead:

```text
Dataset Version
      +
Base Model Version
      +
Training Code Version
      +
Training Configuration
      +
Model Artifact
      +
Evaluation Results
      ↓
   Model Version
```

---

# 1. What should a model version contain?

Suppose we create:

```text
customer-support-model v1.2.0
```

Its metadata should include:

```json
{
  "model_name": "customer-support-model",
  "model_version": "1.2.0",

  "base_model": "llama-3.1-8b",
  "base_model_revision": "abc123",

  "dataset_name": "customer-support",
  "dataset_version": "v5",
  "dataset_fingerprint": "a84bcf7e...",

  "training_code_version": "git-commit-sha",
  "training_config_version": "training-config-v3",

  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  },

  "metrics": {
    "validation_loss": 1.21,
    "instruction_score": 0.91,
    "hallucination_rate": 0.03
  },

  "status": "production"
}
```

This gives us **reproducibility and lineage**.

---

# 2. Model versioning architecture

```text
                  Git Repository
                       │
                       │ Code Version
                       ▼
                 Training Pipeline
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
      Dataset      Configuration   Base Model
      Version        Version        Version
          │            │             │
          └────────────┼─────────────┘
                       ▼
                    Training
                       │
                       ▼
                 Model Artifact
                       │
                       ▼
                 Evaluation
                       │
                       ▼
                 Model Registry
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          Staging  Production  Archived
```

---

# 3. Use immutable versions

Never overwrite a production model.

Bad:

```text
models/
    support-model/
        model.safetensors
```

You don't know what changed.

Better:

```text
models/
│
└── customer-support/
    │
    ├── v1.0.0/
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   ├── metadata.json
    │   └── evaluation.json
    │
    ├── v1.1.0/
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   ├── metadata.json
    │   └── evaluation.json
    │
    └── v2.0.0/
```

Once:

```text
v1.1.0
```

is registered, don't modify it.

If something changes:

```text
Create v1.1.1
```

---

# 4. Use semantic versioning

A useful strategy is:

```text
MAJOR.MINOR.PATCH
```

Example:

```text
1.0.0
```

## Major version

Large behavioral or architectural change.

```text
1.0.0 → 2.0.0
```

Example:

```text
Base Model:
Llama 3.1 → New architecture/model family
```

## Minor version

New capability or improved fine-tuning.

```text
1.0.0 → 1.1.0
```

Example:

```text
New domain training data
New customer-support capability
```

## Patch version

Small correction.

```text
1.1.0 → 1.1.1
```

Example:

```text
Fixed bad training examples
Small LoRA configuration correction
```

The exact versioning rules should be defined by the organization.

---

# 5. Create a model metadata schema

Use Pydantic to enforce consistent metadata.

```python
# app/registry/schema.py

from pydantic import BaseModel
from datetime import datetime


class ModelMetadata(BaseModel):

    model_name: str

    model_version: str

    base_model: str

    base_model_revision: str

    dataset_version: str

    dataset_fingerprint: str

    training_code_version: str

    created_at: datetime

    metrics: dict[str, float]

    status: str
```

Now create metadata:

```python
from datetime import datetime


metadata = ModelMetadata(

    model_name="customer-support",

    model_version="1.2.0",

    base_model="llama-3.1-8b",

    base_model_revision="abc123",

    dataset_version="v5",

    dataset_fingerprint="a84bcf7e123",

    training_code_version="git-sha-xyz",

    created_at=datetime.utcnow(),

    metrics={

        "validation_loss": 1.21,

        "instruction_score": 0.91,

        "hallucination_rate": 0.03
    },

    status="candidate"
)
```

---

# 6. Save model metadata

```python
import json


def save_model_metadata(
    metadata: ModelMetadata,
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

Usage:

```python
save_model_metadata(
    metadata,
    "./models/customer-support/v1.2.0/metadata.json"
)
```

Now the model artifact is not separated from its metadata.

---

# 7. Version LoRA models properly

With LoRA, the complete model may not be saved.

Instead:

```text
Base Model
     +
LoRA Adapter
```

So your model registry must track both.

```text
Model Version: support-v1.2

Base Model:
llama-base-v3

Base Model Revision:
abc123

LoRA Adapter:
adapter-v1.2
```

Save the adapter:

```python
trainer.save_model(
    "./artifacts/support-v1.2"
)

tokenizer.save_pretrained(
    "./artifacts/support-v1.2"
)
```

Artifacts:

```text
support-v1.2/
│
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── metadata.json
└── evaluation.json
```

---

# 8. Create a model registry abstraction

A simple implementation:

```python
# app/registry/model_registry.py

from pathlib import Path
import json


class ModelRegistry:

    def __init__(
        self,
        root: str
    ):

        self.root = Path(root)


    def register_model(
        self,
        model_name: str,
        version: str,
        metadata: dict
    ):

        model_path = (
            self.root
            / model_name
            / version
        )

        model_path.mkdir(
            parents=True,
            exist_ok=False
        )

        metadata_path = (
            model_path
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                default=str
            )

        return str(
            model_path
        )
```

Important:

```python
exist_ok=False
```

Why?

Because we don't want to accidentally overwrite:

```text
v1.2.0
```

---

# 9. Register a model

```python
registry = ModelRegistry(
    root="./models"
)

model_path = registry.register_model(

    model_name=
        "customer-support",

    version="1.2.0",

    metadata=
        metadata.model_dump(
            mode="json"
        )
)

print(model_path)
```

Output:

```text
models/customer-support/1.2.0
```

---

# 10. Model lifecycle stages

A model should have a lifecycle.

```text
              Training
                  │
                  ▼
              Candidate
                  │
                  ▼
              Evaluation
                  │
          ┌───────┴────────┐
          ▼                ▼
        Failed            Passed
                           │
                           ▼
                       Staging
                           │
                           ▼
                      Production
                           │
                           ▼
                       Archived
```

Example states:

```python
from enum import Enum


class ModelStage(str, Enum):

    CANDIDATE = "candidate"

    STAGING = "staging"

    PRODUCTION = "production"

    ARCHIVED = "archived"
```

---

# 11. Promotion based on quality gates

Do not manually say:

```text
"I think this model looks good."
```

Use automated quality gates.

```python
def passes_quality_gate(
    metrics: dict
) -> bool:

    if metrics[
        "instruction_score"
    ] < 0.85:

        return False


    if metrics[
        "hallucination_rate"
    ] > 0.05:

        return False


    if metrics[
        "validation_loss"
    ] > 1.5:

        return False


    return True
```

Use it:

```python
if passes_quality_gate(
    metadata.metrics
):

    metadata.status = "staging"

else:

    metadata.status = "rejected"
```

---

# 12. Compare candidate with production

This is very important.

Suppose:

```text
Current Production:
support-v1.1

Candidate:
support-v1.2
```

Compare them:

```python
def compare_models(
    production_metrics,
    candidate_metrics
):

    comparison = {

        "instruction_improvement":

            candidate_metrics[
                "instruction_score"
            ]
            -
            production_metrics[
                "instruction_score"
            ],


        "hallucination_change":

            candidate_metrics[
                "hallucination_rate"
            ]
            -
            production_metrics[
                "hallucination_rate"
            ]
    }

    return comparison
```

Example:

```text
Metric                   v1.1       v1.2

Instruction Score        0.86       0.91
Hallucination Rate       0.05       0.03
Latency                  1.1 sec    1.2 sec
```

Decision:

```text
v1.2 is better
     ↓
Promote to staging
```

---

# 13. Model registry database design

In a production system, I might store metadata in PostgreSQL.

```sql
CREATE TABLE model_versions (

    id UUID PRIMARY KEY,

    model_name VARCHAR(255),

    version VARCHAR(50),

    base_model VARCHAR(255),

    dataset_version VARCHAR(255),

    dataset_fingerprint TEXT,

    code_version VARCHAR(255),

    artifact_uri TEXT,

    stage VARCHAR(50),

    created_at TIMESTAMP,

    UNIQUE(model_name, version)
);
```

Metrics:

```sql
CREATE TABLE model_metrics (

    id UUID PRIMARY KEY,

    model_version_id UUID,

    metric_name VARCHAR(255),

    metric_value FLOAT,

    FOREIGN KEY (
        model_version_id
    )
    REFERENCES model_versions(id)
);
```

This lets us query:

```sql
SELECT
    version,
    dataset_version,
    stage

FROM model_versions

WHERE model_name =
    'customer-support';
```

---

# 14. Version artifacts in object storage

Production models should generally not live only on a local server.

Conceptually:

```text
Object Storage

models/
│
└── customer-support/
    │
    ├── 1.0.0/
    │   ├── model/
    │   ├── metadata.json
    │   └── metrics.json
    │
    └── 1.1.0/
        ├── model/
        ├── metadata.json
        └── metrics.json
```

The metadata database stores:

```text
Model Name
      +
Version
      +
Artifact URI
      +
Dataset Version
      +
Metrics
      +
Stage
```

---

# 15. Rollback

Versioning makes rollback easy.

Suppose:

```text
Production:

v1.1
```

Deploy:

```text
v1.2
```

Then errors increase.

```text
v1.1
Error Rate: 0.5%

v1.2
Error Rate: 8%
```

Rollback:

```text
Production
     │
     ▼
v1.2 ❌
     │
     ▼
Rollback
     │
     ▼
v1.1 ✅
```

Code:

```python
def rollback(
    registry,
    model_name,
    previous_version
):

    registry.set_stage(

        model_name=model_name,

        version=previous_version,

        stage="production"
    )
```

The main advantage is that you don't retrain during an outage.

You simply deploy the previous known-good artifact.

---

# 16. Canary deployment

Instead of immediately changing:

```text
100% → v1.1
```

to:

```text
100% → v1.2
```

use:

```text
90% → v1.1

10% → v1.2
```

Monitor:

```text
            ┌─── v1.1 → 90%
Requests ───┤
            └─── v1.2 → 10%
```

Metrics:

```text
Latency

Error Rate

User Satisfaction

Hallucination Rate

Safety Violations
```

If v1.2 performs well:

```text
50% → v1.2
```

Then:

```text
100% → v1.2
```

---

# 17. Track model lineage

The most important question is:

> How was this model created?

A model lineage record:

```python
lineage = {

    "model_version":
        "support-v1.2",

    "parent_model":
        "llama-3.1-8b",

    "dataset_version":
        "customer-support-v5",

    "dataset_fingerprint":
        "abc123",

    "training_code_commit":
        "git-sha-xyz",

    "training_config": {

        "learning_rate": 2e-4,

        "epochs": 3,

        "batch_size": 8,

        "lora_r": 16
    }
}
```

Graphically:

```text
Base Model
    │
    ▼
Dataset v5
    │
    ▼
Training Code Commit abc123
    │
    ▼
Training Config v3
    │
    ▼
Model support-v1.2
    │
    ▼
Production Deployment
```

This is **model lineage**.

---

# 18. Using MLflow conceptually

A typical experiment workflow:

```text
Training Run
     │
     ├── Parameters
     │
     ├── Metrics
     │
     ├── Dataset Version
     │
     └── Model Artifact
              │
              ▼
         Model Registry
              │
              ▼
           Version 12
```

Example:

```python
import mlflow


with mlflow.start_run():

    mlflow.log_param(
        "learning_rate",
        2e-4
    )

    mlflow.log_param(
        "dataset_version",
        "v5"
    )

    mlflow.log_metric(
        "validation_loss",
        1.21
    )

    mlflow.log_metric(
        "instruction_score",
        0.91
    )

    mlflow.log_metric(
        "hallucination_rate",
        0.03
    )

    mlflow.log_artifacts(
        "./artifacts"
    )
```

The experiment run captures the evidence behind the model.

---

# 19. Complete production model registration pipeline

```python
def register_trained_model(
    model_name: str,
    version: str,
    dataset_metadata: dict,
    training_config: dict,
    metrics: dict
):

    # 1. Validate metrics
    if not passes_quality_gate(
        metrics
    ):
        raise ValueError(
            "Model failed quality gate"
        )

    # 2. Build metadata
    metadata = {

        "model_name":
            model_name,

        "version":
            version,

        "dataset_version":
            dataset_metadata["version"],

        "dataset_fingerprint":
            dataset_metadata[
                "fingerprint"
            ],

        "training_config":
            training_config,

        "metrics":
            metrics,

        "stage":
            "candidate"
    }

    # 3. Register immutable version
    registry = ModelRegistry(
        "./models"
    )

    registry.register_model(

        model_name,

        version,

        metadata
    )

    return metadata
```

Usage:

```python
register_trained_model(

    model_name=
        "customer-support",

    version="1.2.0",

    dataset_metadata={
        "version": "v5",
        "fingerprint": "abc123"
    },

    training_config={
        "learning_rate": 2e-4,
        "epochs": 3,
        "lora_r": 16
    },

    metrics={
        "validation_loss": 1.2,
        "instruction_score": 0.91,
        "hallucination_rate": 0.03
    }
)
```

---

# 20. Production model versioning flow

```text
                     Dataset v5
                         │
                         ▼
                    Training Run
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
         Code SHA     Config      Base Model
              │          │           │
              └──────────┼───────────┘
                         ▼
                    Model Artifact
                         │
                         ▼
                    Evaluation
                         │
                         ▼
                   Quality Gate
                    │        │
                  Fail      Pass
                    │        │
                    ▼        ▼
                 Reject   Register
                              │
                              ▼
                        support-v1.2
                              │
                              ▼
                           Staging
                              │
                              ▼
                        Canary Deploy
                              │
                              ▼
                         Production
```

---

# Interview-ready answer

> **I version models as immutable artifacts in a model registry. Every model version is linked to the exact base model and revision, dataset version and fingerprint, training code commit, training configuration, hyperparameters, evaluation metrics, and artifact location.**
>
> **I never overwrite an existing production model. Instead, each model receives a unique immutable version, such as semantic versions or registry-generated versions. After training, the model goes through automated quality gates and is registered as a candidate. It is then promoted through staging and production, typically using canary deployment.**
>
> **I maintain model lineage so I can reproduce any model and trace exactly how it was trained. I also keep previous approved versions available for immediate rollback. Tools such as MLflow can track experiments, artifacts, and model registry stages.**

## The key concept

```text
Model Version
    ≠
Just model weights
```

A production model version is:

```text
Model Weights / LoRA Adapter
        +
Base Model Version
        +
Dataset Version
        +
Dataset Fingerprint
        +
Training Code Version
        +
Configuration
        +
Evaluation Metrics
        +
Artifact URI
        +
Deployment Stage
```

That combination gives you **reproducibility, traceability, safe deployment, auditing, and rollback**, which is what proper production model versioning is about.
