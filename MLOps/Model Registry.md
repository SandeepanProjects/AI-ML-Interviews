# MLOps at Staff AI Engineer Level: Model Registry, Model Versioning, Data Versioning, and Experiment Tracking

In a production AI system, training a model is only one part of the problem. A staff-level AI engineer must ensure that every model can be traced to the data, code, parameters, evaluation results, and approval process that produced it.

Imagine you're building an enterprise financial market intelligence platform. You train a model to predict market risk, and a new version performs better in offline testing. Before deploying it, you need to answer:

- Which dataset and dataset snapshot trained this model?
- Which source-code commit and hyperparameters were used?
- How does this model compare with previous experiments?
- Which exact model artifact is running in production?
- Who approved it, and can you roll back safely?
- Can you reproduce the model six months later?

These four MLOps capabilities work together to answer those questions.

## The four pillars

1\. Experiment tracking

Records what happened during each training run: parameters, metrics, code, artifacts and results.

2\. Data versioning

Identifies the exact data snapshot, features and preprocessing inputs used for training and evaluation.

3\. Model versioning

Keeps identifiable, immutable versions of trained model artifacts and their associated metadata.

4\. Model registry

Provides a governed catalog of model versions, their lifecycle status, approvals and deployment references.

A common tool in this space is MLflow. Other options include cloud-managed ML platforms and data-versioning tools such as DVC. The important thing is to understand the architecture and guarantees, not just the product names.

## 1. Experiment tracking

### What problem does it solve?

Suppose you train five models with different hyperparameters:

| Run   | Learning rate | Max depth | F1 score |
| ----- | ------------- | --------- | -------- |
| Run 1 | 0.01          | 4         | 0.78     |
| Run 2 | 0.05          | 4         | 0.81     |
| Run 3 | 0.10          | 6         | 0.76     |
| Run 4 | 0.03          | 8         | 0.84     |
| Run 5 | 0.03          | 10        | 0.82     |

Without experiment tracking, you may have a collection of notebooks, terminal output and model files, but no reliable record of which run produced which result.

Experiment tracking gives every run a persistent identity and records:

- Parameters: learning rate, batch size, model type, number of estimators.
- Metrics: F1, precision, recall, ROC-AUC, latency, cost.
- Artifacts: trained models, plots, evaluation reports and feature-importance files.
- Lineage: source-code commit, dataset version, dependency versions and training configuration.
- Execution metadata: timestamps, status, environment and run ID.

### Production-style code using MLflow

Install:

```
pip install mlflow scikit-learn pandas
```

Create `train.py`:

```
import hashlibimport jsonimport osimport subprocessimport mlflowimport mlflow.sklearnfrom sklearn.datasets import load_breast_cancerfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import (    accuracy_score,    f1_score,    precision_score,    recall_score,)from sklearn.model_selection import train_test_splitmlflow.set_experiment("financial-risk-classification")# In production, load a governed, versioned dataset here.dataset = load_breast_cancer(as_frame=True)X = dataset.datay = dataset.targetX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=42,    stratify=y,)params = {    "n_estimators": 200,    "max_depth": 8,    "random_state": 42,}try:    git_commit = subprocess.check_output(        ["git", "rev-parse", "HEAD"],        text=True,        stderr=subprocess.DEVNULL,    ).strip()except (subprocess.SubprocessError, FileNotFoundError):    git_commit = "unknown"with mlflow.start_run(run_name="random-forest-v1") as run:    model = RandomForestClassifier(**params)    model.fit(X_train, y_train)    predictions = model.predict(X_test)    metrics = {        "accuracy": accuracy_score(y_test, predictions),        "f1": f1_score(y_test, predictions),        "precision": precision_score(y_test, predictions),        "recall": recall_score(y_test, predictions),    }    mlflow.log_params(params)    mlflow.log_metrics(metrics)    mlflow.set_tags({        "git_commit": git_commit,        "dataset_name": "breast_cancer_demo",        "dataset_version": "sklearn_builtin",        "model_family": "random_forest",    })    mlflow.log_dict(        {"features": list(X.columns)},        "feature_schema.json",    )    mlflow.sklearn.log_model(        sk_model=model,        name="model",        input_example=X_train.head(3),    )    print("Run ID:", run.info.run_id)    print("Metrics:", json.dumps(metrics, indent=2))
```

Start the MLflow UI in another terminal:

```
mlflow server --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` and inspect your experiment runs. For a team, configure a shared tracking server, database-backed metadata storage and durable artifact storage rather than relying on an individual developer's local setup.

Note: This is an illustrative classification experiment using a built-in dataset, not a validated financial model. In a real training pipeline, the train/test split, feature engineering, evaluation strategy and dataset identity must reflect the actual business problem.

### What happens internally?

Training code

MLflow tracking API

Metadata store

Run IDs, parameters, metrics, tags

Artifact store

Model files, plots, reports

The metadata store and artifact store have different responsibilities. Metrics can live in a database, while large model files live in object storage. A run ID connects them.

Staff-level consideration: a high F1 score does not automatically make a model production-ready. You must consider leakage, temporal validation, class imbalance, calibration, subgroup performance, inference latency, model size, and business cost.

## 2. Data versioning

### What problem does it solve?

A model's performance depends on more than its algorithm. It depends on the exact data used to train and evaluate it.

Consider this sequence:

- Monday: train on 2 million transaction records.
- Tuesday: fix incorrect labels and remove duplicates.
- Wednesday: train again on 2.1 million records.

Even with identical code and hyperparameters, Wednesday's model can differ because its training data changed.

Data versioning makes it possible to identify the precise data snapshot and reproduce the transformation pipeline.

### What should be versioned?

| Component         | Example                                              |
| ----------------- | ---------------------------------------------------- |
| Raw data snapshot | Object-storage path plus immutable snapshot ID       |
| Labels            | Labeling rules and label-table version               |
| Features          | Feature definitions and transformation version       |
| Dataset split     | Train/validation/test membership                     |
| Schema            | Column names, types and constraints                  |
| Data quality      | Missing values, duplicates, range checks             |
| Provenance        | Source system, ingestion timestamp and snapshot hash |

### Important distinction: data version versus data location

A path such as `s3://company-data/training.csv` does not necessarily identify a version. If someone overwrites that object, the same path may point to different contents later.

Prefer immutable object versions, snapshot IDs or content hashes, backed by access controls and retention policies.

### A simple content hash

```
import hashlibfrom pathlib import Pathdef sha256_file(path: str) -> str:    digest = hashlib.sha256()    with Path(path).open("rb") as file:        for chunk in iter(lambda: file.read(1024 * 1024), b""):            digest.update(chunk)    return digest.hexdigest()dataset_path = "data/training.csv"dataset_hash = sha256_file(dataset_path)print("Dataset SHA-256:", dataset_hash)
```

This detects changes to the exact file bytes. It does not by itself store the file, make it immutable, or explain its business meaning. For a large multi-file dataset, create a manifest of file paths, hashes, schema, row counts and transformation versions. Ensure the manifest uses a stable ordering and canonical representation.

### Data versioning with DVC

DVC can track large data files while Git tracks code and small metadata files.

A typical workflow:

```
pip install dvc

git init
dvc init

dvc add data/training.csv

git add data/training.csv.dvc .gitignore .dvc/
git commit -m "Version training dataset"

# After configuring an appropriate remote:
dvc remote add -d storage s3://my-ml-bucket/dvc-store
dvc push
```

The `.dvc` metadata file records the tracked artifact's identity; the actual data is stored in DVC's cache and remote. For collaboration, configure authentication and permissions securely. Never commit cloud credentials.

To retrieve a previous dataset version, check out the Git commit containing its DVC metadata and run:

```
dvc pull
```

This restores the corresponding data artifacts if they remain available in the configured remote.

### Staff-level data-versioning design

For a production pipeline, I would record a training manifest like this:

```
{
  "dataset_id": "market-risk-training",
  "dataset_version": "2026-10-15",
  "source_snapshot": "warehouse-snapshot-7842",
  "content_manifest_sha256": "example-hash",
  "schema_version": "v3",
  "feature_pipeline_version": "git:abc123",
  "label_definition_version": "risk-label-v2",
  "split_strategy": "time_based",
  "created_at": "2026-10-15T03:00:00Z"
}
```

The example hash is a placeholder. A real pipeline should calculate and validate it rather than hard-code it.

For time-series financial data, a random split can leak future information into training. A time-based split with appropriate label windows, point-in-time features and controls for look-ahead bias is often more appropriate.

## 3. Model versioning

### What exactly is a model version?

A model version is an identifiable release of a trained model artifact.

For example:

```
financial-risk-classifier
├── version 1  → trained with dataset snapshot A
├── version 2  → trained with corrected labels
├── version 3  → new features and tuned hyperparameters
└── version 4  → retrained on recent data
```

Each version should be traceable to the exact artifact, training run, dataset, code and evaluation results that produced it.

A critical distinction:

- Code versioning: versions the Python code and configuration, usually using Git.
- Data versioning: versions the training and evaluation inputs.
- Model versioning: versions the resulting trained artifacts.
- Model registry: catalogs and governs these model versions.

A Git commit alone cannot identify a trained model, because the same code can produce different artifacts when the data, dependencies, random seeds, hardware or nondeterministic operations change.

### How should a model version be identified?

Use an immutable artifact reference and a cryptographic hash where practical.

For example:

```
Model name: financial-risk-classifier
Model version: 17
Artifact URI: s3://ml-artifacts/sha256/ab12.../model/
Artifact SHA-256: ab12...
Training run ID: 9f81...
Dataset version: market-risk-2026-10-15
Git commit: 7cd34a...
Evaluation suite: risk-eval-v8
```

The version number is a human-friendly identifier. The artifact hash helps verify that the stored bytes have not changed. The run ID and dataset version explain how that artifact was produced.

### A simplified model manifest

In production, I would associate each model artifact with a manifest:

```
from dataclasses import dataclass@dataclass(frozen=True)class ModelManifest:    model_name: str    model_version: str    artifact_uri: str    artifact_sha256: str    training_run_id: str    dataset_version: str    git_commit: str    evaluation_suite_version: str
```

`frozen=True` prevents ordinary reassignment of the dataclass fields. It does not make the artifact, database record or storage object immutable. Enforce immutability with storage policies, permissions and registry controls.

### Versioning a model with MLflow

MLflow supports model registration and version tracking. The following example assumes you have already logged a model under the `model` artifact name in a completed run.

```
import mlflowfrom mlflow import MlflowClientclient = MlflowClient()model_name = "financial-risk-classifier"run_id = "YOUR_COMPLETED_RUN_ID"model_uri = f"runs:/{run_id}/model"# Create the registered model if it does not already exist.try:    client.create_registered_model(model_name)except Exception:    # In production, catch and verify only the specific    # "already exists" condition for your MLflow version.    if not client.get_registered_model(model_name):        raise# Register this run's model artifact as a new model version.registered = mlflow.register_model(    model_uri=model_uri,    name=model_name,)print("Registered model:", registered.name)print("Model version:", registered.version)
```

For production code, avoid broad exception handling as a substitute for checking a known registry conflict. Also, use the registry's configured permissions and artifact access controls.

Load an exact version when you need reproducibility:

```
import mlflow.pyfuncmodel_uri = "models:/financial-risk-classifier/17"model = mlflow.pyfunc.load_model(model_uri)predictions = model.predict(input_data)
```

This is preferable to loading an unqualified mutable reference when you need to reproduce a particular deployment.

### What makes a version genuinely reproducible?

A staff-level engineer checks more than the model file:

1. Artifact integrity: Is the model artifact the expected one?
2. Dependency compatibility: Can the serving environment load it?
3. Feature compatibility: Does inference produce the same input schema the model expects?
4. Data lineage: Is the exact training snapshot recorded and accessible?
5. Evaluation lineage: Are the test dataset, metric definitions and evaluation code recorded?
6. Runtime compatibility: Are framework versions and any required custom code available?
7. Operational readiness: Has the model passed security, latency and deployment checks?

Model files may contain serialized executable objects or custom code. Load only trusted artifacts, restrict artifact write permissions, and use safe serialization formats and formats supported by your serving stack.

## 4. Model registry

### What problem does it solve?

A registry is the governed catalog for models and their versions.

Experiment tracking answers:

> Which training run produced which metrics and artifacts?

The model registry answers:

> Which model versions are available, which are approved, and which version is allowed to be deployed?

A registry commonly stores or references:

- Model name and version
- Artifact URI and artifact identity
- Training run ID
- Evaluation results and model lineage
- Input/output schema or signature
- Model owner and relevant tags
- Lifecycle status, approval information and deployment references
- Audit history, depending on the platform

The registry should point to model artifacts rather than require every artifact to be stored directly in the registry's metadata database.

### Registry lifecycle

1\. Candidate registered

Artifact and lineage recorded

2\. Evaluation and validation

Quality, safety, fairness where relevant, latency and compatibility

3\. Approval gate

Automated policy checks and authorized human approval where required

4\. Deployment

Canary, shadow, or controlled rollout

5\. Monitor or roll back

Monitor production behavior and retain the previous known-good version

The lifecycle labels are conceptual; the exact stages and available features depend on the registry implementation. For example, MLflow supports registered model versions and aliases. Older MLflow workflows may use model stages, while newer workflows can use aliases such as `champion` and `challenger`.

### Why use aliases?

Suppose version 17 is serving production traffic. You can point a logical alias at it:

```
financial-risk-classifier
    champion   → version 17
    challenger → version 18
```

After version 18 passes the required checks, you can update the `champion` alias to point to version 18.

The application resolves the alias to choose a model, but the alias is mutable. For audit and incident investigation, log both the alias and the exact resolved model version for each deployment or inference session, as appropriate.

### Example: using an MLflow alias

```
from mlflow import MlflowClientclient = MlflowClient()model_name = "financial-risk-classifier"# Assign an alias after validation and approval.client.set_registered_model_alias(    name=model_name,    alias="champion",    version="18",)# A rollback can move the alias to a previous version.client.set_registered_model_alias(    name=model_name,    alias="champion",    version="17",)
```

These calls update registry metadata; they do not automatically deploy a new serving container, update a Kubernetes Deployment, or roll back traffic. Your deployment system must resolve the approved version and perform that action.

### Registry versus deployment platform

They are related but different.

| Component             | Responsibility                                   |
| --------------------- | ------------------------------------------------ |
| Experiment tracker    | Records training runs and results                |
| Model registry        | Catalogs versions and their governance metadata  |
| Artifact/object store | Holds model files and related artifacts          |
| CI/CD pipeline        | Validates, packages and releases the model       |
| Serving platform      | Loads the artifact and handles inference traffic |
| Monitoring system     | Measures live quality, latency, errors and drift |

For example, a registry may designate version 18 as the approved candidate. A CI/CD pipeline then builds a serving image, deploys it to staging, runs smoke tests and rolls it out to production.

A registry does not, by itself, guarantee that the model is safe, accurate or currently deployed.

## 5. How all four work together in a real MLOps pipeline

Let's put these concepts together for a financial risk classification system.

Step 1 — Data pipeline

Ingest, validate, transform and snapshot data

Step 2 — Experiment tracking

Record parameters, metrics, code commit and run ID

Step 3 — Model artifact creation

Save artifact, hash and training manifest

Step 4 — Registry

Register candidate and associate its lineage

Step 5 — Evaluation and approval

Compare with baseline, enforce release criteria

Step 6 — Deployment

Release the exact approved version

Step 7 — Production monitoring

Monitor quality, drift, latency, failures and business outcomes

### A practical release gate

A staff engineer should define explicit release criteria rather than simply deploy the model with the highest validation score.

For example:

```
def validate_candidate(metrics: dict, baseline: dict) -> None:    """Illustrative release policy for a classification model."""    if metrics["f1"] < 0.85:        raise ValueError("F1 score is below the minimum threshold")    if metrics["recall"] < 0.90:        raise ValueError("Recall is below the minimum threshold")    if metrics["p95_latency_ms"] > 100:        raise ValueError("P95 latency exceeds the serving budget")    if metrics["f1"] < baseline["f1"] - 0.01:        raise ValueError("Candidate regresses too far against baseline")
```

These are illustrative thresholds, not universal standards. The business, risk team and model owners should define the thresholds based on the impact of false positives and false negatives, validation methodology, and production constraints.

A complete release gate would also validate artifact integrity, schema compatibility, reproducibility metadata, data-quality checks and any required security or compliance approvals.

## 6. How to implement the architecture in an enterprise project

For a production platform, I would separate responsibilities into modules rather than put everything inside one training script.

```
mlops/
├── data/
│   ├── ingestion.py
│   ├── validation.py
│   ├── snapshot.py
│   └── manifest.py
├── training/
│   ├── train.py
│   ├── config.py
│   └── pipeline.py
├── tracking/
│   └── mlflow_tracker.py
├── evaluation/
│   ├── metrics.py
│   ├── regression_checks.py
│   └── release_gate.py
├── registry/
│   ├── client.py
│   └── promotion.py
├── deployment/
│   ├── build.py
│   └── release.py
└── monitoring/
    ├── drift.py
    └── quality.py
```

The call sequence would be:

1. `snapshot.py` identifies and records the training dataset.
2. `pipeline.py` loads the approved snapshot and invokes training.
3. `mlflow_tracker.py` records parameters, metrics, lineage and artifacts.
4. `metrics.py` evaluates the candidate on the correct evaluation dataset.
5. `release_gate.py` checks the release policy.
6. `promotion.py` registers or approves the model version and updates the relevant registry metadata.
7. `release.py` deploys the exact approved version.
8. `quality.py` and `drift.py` monitor production behavior.

The modules above are a recommended design, not code from an existing repository.

### Where do Git, DVC, MLflow and the registry fit?

| Technology                                  | Primary purpose                                          |
| ------------------------------------------- | -------------------------------------------------------- |
| Git                                         | Source code, configuration and pipeline definitions      |
| DVC or a governed data platform             | Dataset and data-artifact versioning                     |
| MLflow Tracking                             | Experiments, metrics, parameters and artifacts           |
| MLflow Model Registry or a managed registry | Model catalog, versions, aliases and governance metadata |
| Object storage                              | Durable model and data artifacts                         |
| CI/CD                                       | Automated validation and deployment                      |
| Monitoring stack                            | Live inference performance and model-quality signals     |

These tools can overlap in functionality. For example, a managed ML platform may provide dataset lineage, experiment tracking, a registry and deployment integration in one service. You still need to understand the guarantees each capability provides.

## 7. Common production mistakes

Mistake 1: Tracking metrics but not data lineage

You know a model scored 0.91 F1, but cannot identify the dataset or label definition used. Record a stable dataset snapshot ID and preprocessing version.

Mistake 2: Overwriting model files

A storage path changes while the registry still points to it. Use immutable artifact locations, content hashes, restricted writes and retention policies.

Mistake 3: Confusing the best experiment with the production model

A candidate with the highest validation metric may fail latency, security or compatibility requirements. Separate model selection from release approval.

Mistake 4: Using a mutable alias as the only audit reference

An alias such as `champion` can point to a different version tomorrow. Record the resolved version and deployment identifier.

Mistake 5: Assuming a version number guarantees reproducibility

Reproduction may fail because dependencies, feature code, data or runtime configuration changed. Capture those dependencies and verify that the referenced artifacts remain available.

## 8. Staff-level interview answer

A strong answer to “How do you manage models in production using MLOps?” could be:

> I separate experiment tracking, data versioning, model versioning and model registry because they solve different problems.
>
> I use experiment tracking to capture training parameters, evaluation metrics, artifacts and execution metadata. I version the datasets, feature transformations and label definitions so every training run can be traced to its inputs. Each trained model is stored as an identifiable artifact with integrity information and a link to its training run.
>
> I register the artifact in a model registry, where we maintain version metadata, evaluation results and lifecycle governance. Before promotion, a release gate verifies model quality, baseline regression, schema compatibility, latency, security and any required approvals. The deployment pipeline then releases the exact approved version using a controlled rollout strategy.
>
> In production, we monitor service health, latency, data drift and model quality where labels are available. If the new version violates operational or business thresholds, we can roll back to a previous known-good version. For auditability, we retain the resolved model version, dataset snapshot, source commit and deployment metadata.
>
> My primary goal is not just to train an accurate model; it is to make the model reproducible, traceable, deployable, observable and safely reversible.

## 9. Test your understanding

1\. Your model achieved 95% accuracy, but nobody knows which dataset trained it. Which capability is missing?

Experiment tracking only

Reliable data versioning and lineage

Model registry only

2\. You need to compare 100 runs with different learning rates and F1 scores. What is the primary tool?

Git

Experiment tracking

Deployment platform

3\. Production model version 18 is faulty. What enables a controlled rollback?

Retrain the model from scratch

A retained, known-good model artifact plus deployment/registry controls

The original accuracy score alone

Check answers

The core principle to remember: experiment tracking tells you what happened, data versioning tells you what data was used, model versioning identifies what was produced, and the registry governs which model versions are available and approved. Together, they form the lineage and release-control foundation of an MLOps platform.