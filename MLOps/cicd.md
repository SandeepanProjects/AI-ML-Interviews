CI/CD for Machine Learning is commonly called **MLOps CI/CD** because it automates not only application deployment but also **data validation, model training, model testing, deployment, monitoring, and retraining**.

Unlike traditional software, ML systems have **three versioned artifacts**:

* **Code**
* **Data**
* **Model**

A production pipeline must manage all three.

---

# Traditional CI/CD vs ML CI/CD

| Traditional Software | ML Systems                                      |
| -------------------- | ----------------------------------------------- |
| Source code          | Source code + Data + Models                     |
| Unit tests           | Unit tests + Data validation + Model evaluation |
| Build application    | Train model + Build application                 |
| Deploy application   | Register model + Deploy model                   |
| Monitor application  | Monitor application + Data drift + Model drift  |

---

# End-to-End MLOps CI/CD Pipeline

```text
                    Git Push
                        │
                        ▼
                 Continuous Integration
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    Lint Code      Unit Tests     Data Validation
                        │
                        ▼
                  Train Model
                        │
                        ▼
                Evaluate Metrics
                        │
        Accuracy > Threshold?
               │
        ┌──────┴──────┐
        │             │
       No            Yes
        │             │
     Stop Build       ▼
                 Register Model
                        │
                        ▼
                Build Docker Image
                        │
                        ▼
             Push Image to Registry
                        │
                        ▼
         Deploy to Staging Kubernetes
                        │
                        ▼
                Integration Tests
                        │
                        ▼
            Canary / Blue-Green Deploy
                        │
                        ▼
             Production Deployment
                        │
                        ▼
         Monitoring & Drift Detection
                        │
                        ▼
              Automatic Retraining
```

---

# Step 1: Source Control

Keep everything in Git.

```text
ml-project/

├── data/
├── notebooks/
├── src/
├── models/
├── tests/
├── Dockerfile
├── requirements.txt
└── .github/workflows/
```

Avoid committing large datasets directly into Git. Instead, use object storage (S3, GCS, Azure Blob) or a data versioning tool like DVC.

---

# Step 2: Continuous Integration (CI)

Whenever a developer pushes code:

```text
Developer

↓

Git Push

↓

GitHub Actions

↓

Run Pipeline
```

Typical CI tasks:

* Code formatting (`black`, `ruff`)
* Linting
* Unit tests (`pytest`)
* Security scanning
* Dependency checks
* Data validation
* Small inference smoke tests

Example GitHub Actions workflow:

```yaml
name: ML CI

on:
  push:
    branches:
      - main

jobs:
  test:

    runs-on: ubuntu-latest

    steps:

      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5

      - run: pip install -r requirements.txt

      - run: pytest

      - run: python validate_data.py
```

---

# Step 3: Data Validation

Bad data produces bad models.

Validate:

* Missing values
* Duplicate rows
* Invalid categories
* Schema changes
* Feature ranges
* Null percentages

Example:

```python
import pandas as pd

df = pd.read_csv("train.csv")

assert "age" in df.columns
assert df["age"].isnull().sum() == 0
assert df["age"].min() >= 0
```

If validation fails, stop the pipeline.

---

# Step 4: Train the Model

```python
model.fit(X_train, y_train)
```

Artifacts produced:

```text
model.pkl

metrics.json

confusion_matrix.png
```

---

# Step 5: Evaluate the Model

Compare with the current production model.

Example:

```python
new_accuracy = 0.94
production_accuracy = 0.92

if new_accuracy > production_accuracy:
    deploy = True
```

Common metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* RMSE
* MAE

---

# Step 6: Register the Model

Store approved models in a registry.

```text
Model Registry

Fraud Model

Version 1

Version 2

Version 3
```

Metadata:

```text
Version

Training dataset

Git commit

Accuracy

Created by

Approval status
```

Popular registries:

* MLflow
* SageMaker Model Registry
* Vertex AI Model Registry
* Azure ML Registry

---

# Step 7: Build the Docker Image

```dockerfile
FROM python:3.12-slim

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn","app:app"]
```

Build:

```bash
docker build -t fraud-api:v3 .
```

---

# Step 8: Deploy to Staging

```text
Developer

↓

Docker Image

↓

Kubernetes

↓

Staging
```

Run:

* API tests
* Load tests
* Integration tests
* Model inference tests

---

# Step 9: Deploy to Production

Prefer gradual rollouts.

## Blue-Green

```text
Blue (Old)

↓

Switch

↓

Green (New)
```

Rollback is immediate by switching traffic back.

## Canary

```text
Users

90%

↓

Old Model

10%

↓

New Model
```

Increase traffic only after observing healthy metrics.

---

# Step 10: Monitor the Model

Monitor:

### Infrastructure

* CPU
* Memory
* GPU
* Disk

### Application

* API latency
* Error rate
* Throughput

### ML

* Prediction distribution
* Data drift
* Model drift
* Accuracy
* Feature drift

Example:

```text
Latency

120 ms

↓

450 ms

Alert
```

---

# Step 11: Automatic Retraining

When drift is detected:

```text
Data Drift

↓

Pipeline Triggered

↓

Retrain Model

↓

Evaluate

↓

Deploy
```

Retraining can also be scheduled (daily/weekly) or event-driven.

---

# Complete Production Architecture

```text
                 GitHub Repository
                        │
                        ▼
                 GitHub Actions CI
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Unit Tests     Data Validation    Security Scan
                        │
                        ▼
                  Train Model
                        │
                        ▼
                 Evaluate Metrics
                        │
                        ▼
                Model Registry
                        │
                        ▼
                Build Docker Image
                        │
                        ▼
              Container Registry
                        │
                        ▼
                Kubernetes Cluster
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      API Pod 1     API Pod 2     API Pod 3
                        │
                        ▼
                 Load Balancer
                        │
                        ▼
                     Clients
                        │
                        ▼
        Prometheus + Grafana + OpenTelemetry
                        │
                        ▼
             Drift Detection & Alerts
```

---

# Recommended Tool Stack

| Stage               | Common Tools                                     |
| ------------------- | ------------------------------------------------ |
| Source Control      | Git, GitHub, GitLab                              |
| CI                  | GitHub Actions, GitLab CI, Jenkins               |
| Data Versioning     | DVC, LakeFS                                      |
| Experiment Tracking | MLflow, Weights & Biases                         |
| Model Registry      | MLflow, SageMaker, Vertex AI                     |
| Containerization    | Docker                                           |
| Image Registry      | Docker Hub, Amazon ECR, Google Artifact Registry |
| Orchestration       | Kubernetes                                       |
| Deployment          | Argo CD, Flux CD                                 |
| Monitoring          | Prometheus, Grafana                              |
| Logging             | ELK Stack, Loki                                  |
| Tracing             | OpenTelemetry                                    |
| Drift Detection     | Evidently AI, Arize AI, WhyLabs                  |

---

# Interview Scenario

**Question:** *How would you implement CI/CD for a fraud detection model?*

A strong answer would cover:

1. Push code to Git.
2. CI runs linting, unit tests, and data validation.
3. Train the model on the latest approved dataset.
4. Compare metrics with the production model.
5. Register the model if it passes quality thresholds.
6. Build a Docker image containing the inference service.
7. Deploy to a staging Kubernetes environment.
8. Run integration and load tests.
9. Perform a canary deployment to production.
10. Monitor latency, error rate, prediction quality, data drift, and model drift.
11. Trigger retraining or roll back automatically if performance degrades.

This demonstrates an understanding of the complete MLOps lifecycle rather than just model serving, which is what interviewers for senior ML/AI engineering roles typically look for.
