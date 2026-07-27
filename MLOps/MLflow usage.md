# MLflow Explained Properly (Production AI Engineer Perspective)

If you're building ML or LLM systems in production, **MLflow** is one of the most important tools you'll use.

Think of MLflow as **Git + Docker + CI/CD + Registry + Experiment Dashboard** for machine learning.

Without MLflow:

```
train_v1.py
train_final.py
train_final_new.py
train_final_latest.py
train_final_latest2.py
```

After two months nobody knows:

* Which model is in production?
* Which dataset was used?
* Which hyperparameters produced it?
* Which code version created it?
* Why accuracy dropped?

MLflow solves these problems.

---

# What problems does MLflow solve?

Suppose you're training a fraud detection model.

You experiment with

```
Random Forest
XGBoost
LightGBM
CatBoost
```

Each experiment has

* different learning rate
* different max depth
* different dataset
* different feature engineering
* different preprocessing

Very quickly you end up with hundreds of experiments.

Example

```
Experiment 1
Accuracy = 89%

Experiment 2
Accuracy = 92%

Experiment 3
Accuracy = 91%

Experiment 4
Accuracy = 94%
```

Six months later your manager asks

> Which model produced 94%?

Without MLflow

```
No idea.
```

With MLflow

```
Everything is saved.
```

---

# MLflow Components

```
                  MLflow

         +----------------------+
         | Experiment Tracking  |
         +----------------------+

         +----------------------+
         | Model Registry       |
         +----------------------+

         +----------------------+
         | Artifact Storage     |
         +----------------------+

         +----------------------+
         | Model Deployment     |
         +----------------------+
```

Each solves a different problem.

---

# 1. Experiment Tracking

This is the most used feature.

During training MLflow records

```
Parameters

Metrics

Artifacts

Source code

Git commit

Environment

Execution time
```

Example

```
Run 101

Learning Rate = 0.001

Batch Size = 64

Epochs = 20

Accuracy = 94%

F1 Score = 0.92

Model Size = 120MB
```

Everything gets stored automatically.

---

# Basic Example

Install

```bash
pip install mlflow
```

Training

```python
import mlflow

mlflow.set_experiment("Fraud Detection")

with mlflow.start_run():

    mlflow.log_param("learning_rate",0.001)
    mlflow.log_param("epochs",20)

    accuracy = 0.94

    mlflow.log_metric("accuracy",accuracy)

    print("Training Complete")
```

Run

```
python train.py
```

Start UI

```
mlflow ui
```

Open

```
http://localhost:5000
```

You'll see

```
Run ID

Parameters

Metrics

Artifacts

Execution Time
```

---

# Logging Parameters

Parameters describe how the model was trained.

```python
mlflow.log_param("batch_size",64)

mlflow.log_param("optimizer","Adam")

mlflow.log_param("dropout",0.3)
```

Dashboard

```
Run

Batch Size

Optimizer

Dropout
```

---

# Logging Metrics

Metrics are values changing during training.

```python
mlflow.log_metric("accuracy",0.91)

mlflow.log_metric("precision",0.92)

mlflow.log_metric("recall",0.90)

mlflow.log_metric("f1",0.91)
```

Dashboard

```
Accuracy

Precision

Recall

F1
```

---

# Logging Multiple Epochs

```python
for epoch in range(20):

    loss = train()

    mlflow.log_metric(
        "loss",
        loss,
        step=epoch
    )
```

MLflow plots

```
Loss

1.2

0.9

0.5

0.2
```

as a graph automatically.

---

# Logging Artifacts

Artifacts include

```
Model

Plots

Confusion Matrix

ROC Curve

Feature Importance

Predictions

CSV

Images

PDF
```

Example

```python
mlflow.log_artifact("confusion_matrix.png")

mlflow.log_artifact("roc_curve.png")
```

Now every run contains plots.

---

# Logging Models

Suppose using sklearn.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X,y)
```

Save

```python
import mlflow.sklearn

mlflow.sklearn.log_model(
    model,
    "fraud_model"
)
```

Now the model is stored inside MLflow.

---

# Complete Training Pipeline

```python
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Iris")

with mlflow.start_run():

    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")
```

---

# Model Registry

Imagine several models:

```
Model A

94%
```

```
Model B

96%
```

```
Model C

97%
```

Which one goes to production?

The registry keeps versions.

```
Fraud Model

Version 1

Version 2

Version 3

Version 4
```

Each version has stages such as:

```
Development

↓

Staging

↓

Production

↓

Archived
```

No manual file renaming is required.

---

# Register Model

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="FraudDetection"
)
```

The registry automatically creates:

```
FraudDetection

Version 1
```

The next registration becomes:

```
Version 2
```

---

# Load Production Model

```python
model = mlflow.pyfunc.load_model(
    "models:/FraudDetection/Production"
)

predictions = model.predict(X_test)
```

Your application always loads the model currently marked as **Production**, regardless of version number.

---

# Artifact Storage

MLflow can store artifacts in:

* Local filesystem
* Amazon S3
* Azure Blob Storage
* Google Cloud Storage
* MinIO
* NFS

Example structure:

```
mlruns/
    experiment/
        run1/
            metrics/
            params/
            artifacts/
                model/
                roc.png
                cm.png
```

---

# Hyperparameter Tuning

Instead of keeping notes manually:

```python
for lr in [0.1, 0.01, 0.001]:
    for depth in [5, 10]:
        with mlflow.start_run():
            ...
```

MLflow automatically records every combination, making comparison easy.

---

# Production Workflow

```
Data

   │

Feature Engineering

   │

Training

   │

MLflow Tracking

   │

Evaluation

   │

Model Registry

   │

Approval

   │

Deployment

   │

Production

   │

Monitoring

   │

Retraining
```

---

# MLflow with CI/CD

A common enterprise pipeline is:

```
Git Push
     │
GitHub Actions
     │
Run Tests
     │
Train Model
     │
Log to MLflow
     │
Evaluate
     │
Register Model
     │
Deploy to Kubernetes
     │
Monitor
```

---

# MLflow with LLM Applications

MLflow is useful beyond classical ML. For Retrieval-Augmented Generation (RAG) and LLM systems, you can log:

* Prompt versions
* LLM provider and model (e.g., GPT-4.1, Llama)
* Temperature, top-p, max tokens
* Retrieval parameters (top-k, chunk size, embedding model)
* Evaluation metrics (faithfulness, answer relevance, latency)
* Prompt templates
* Generated outputs for offline evaluation
* Cost and token usage

Example:

```python
with mlflow.start_run():
    mlflow.log_param("llm_model", "gpt-4.1")
    mlflow.log_param("temperature", 0.2)
    mlflow.log_param("retriever_top_k", 5)

    mlflow.log_metric("faithfulness", 0.94)
    mlflow.log_metric("latency_ms", 820)
    mlflow.log_metric("cost_usd", 0.012)
```

This lets you compare prompt and retrieval strategies just like traditional model experiments.

---

# MLflow Best Practices

* Create a separate experiment for each project (e.g., Fraud Detection, Customer Churn, RAG QA).
* Log all important hyperparameters, even if they don't change often.
* Record multiple evaluation metrics instead of relying on accuracy alone.
* Save trained models, plots, and preprocessing artifacts together.
* Register only validated models in the Model Registry.
* Promote models through Development → Staging → Production after automated evaluation.
* Store artifacts in cloud object storage (S3, GCS, Azure Blob) for production deployments.
* Integrate MLflow with CI/CD pipelines to automate training, evaluation, registration, and deployment.

---

# MLflow in an Enterprise AI Platform

For a production multi-tenant AI platform, MLflow typically fits into the architecture like this:

```text
                GitHub
                   │
              CI/CD Pipeline
                   │
        Train / Fine-tune Model
                   │
             MLflow Tracking
                   │
         Metrics + Parameters
         Models + Artifacts
                   │
          MLflow Model Registry
                   │
         ┌─────────┴─────────┐
         │                   │
    Staging Model      Production Model
         │                   │
         └─────────┬─────────┘
                   │
          FastAPI / vLLM / TGI
                   │
            Kubernetes Cluster
                   │
      Prometheus + Grafana + OpenTelemetry
                   │
      Drift Detection & Automated Retraining
```

In this setup, MLflow becomes the **system of record** for experiments and model versions, while orchestration tools (such as CI/CD, Kubernetes, and monitoring systems) handle deployment, scaling, and observability.

For senior AI engineering interviews, being able to explain how MLflow integrates with training pipelines, model governance, deployment, monitoring, and rollback is often more valuable than simply knowing the API calls.
