Model versioning is the practice of **tracking, storing, and managing every version of a machine learning model** throughout its lifecycle. It enables you to reproduce results, compare models, roll back faulty deployments, audit changes, and support safe experimentation.

Unlike traditional software versioning, ML versioning must manage **code, data, model artifacts, and configuration** together.

---

# Why Model Versioning?

Imagine you deploy a fraud detection model.

```text
January

Model V1

Accuracy = 91%
```

One month later:

```text
February

Model V2

Accuracy = 95%
```

Then users report many false positives.

Without versioning:

* You don't know what changed.
* You can't roll back easily.
* You can't reproduce training.
* Auditing becomes difficult.

With versioning:

```text
Registry

V1 → V2 → V3 → V4
         ↑
      Rollback
```

---

# What Should Be Versioned?

A production model consists of much more than a `.pkl` file.

```text
Model Version

├── Model weights
├── Training dataset version
├── Feature engineering code
├── Hyperparameters
├── Python dependencies
├── Metrics
├── Training logs
├── Git commit SHA
├── Docker image
└── Deployment configuration
```

---

# Semantic Versioning

Many teams use semantic versioning.

```
Major.Minor.Patch
```

Example

```
1.0.0

↓

1.1.0

↓

1.2.0

↓

2.0.0
```

Meaning:

| Version | Meaning                       |
| ------- | ----------------------------- |
| 1.0.0   | Initial production model      |
| 1.1.0   | Better features or retraining |
| 1.1.1   | Bug fix                       |
| 2.0.0   | Major architecture change     |

---

# Strategy 1: Sequential Versioning

The simplest approach.

```text
Fraud Model

V1

↓

V2

↓

V3

↓

V4
```

Each version stores:

```text
Version 3

Accuracy = 95%

Dataset = v5

Git = a8f91c

Created = 2026-07-25
```

Best for:

* Small teams
* Simple pipelines

---

# Strategy 2: Stage-Based Versioning

A model moves through deployment stages.

```text
Training

↓

Staging

↓

Production

↓

Archived
```

Example:

```text
Version 12

Stage = Production
```

```text
Version 13

Stage = Staging
```

This makes promotion safer.

---

# Strategy 3: Champion-Challenger

Common in production ML.

```text
Production

Champion

↓

New Model

Challenger
```

Traffic:

```text
Users

↓

Champion

95%
```

```text
Users

↓

Challenger

5%
```

Compare:

* Accuracy
* Latency
* Cost
* Business metrics

If the challenger performs better, promote it to champion.

---

# Strategy 4: Canary Deployment

Gradually shift traffic to the new model.

```text
V1

100%
```

↓

```text
V1 90%

V2 10%
```

↓

```text
V1 50%

V2 50%
```

↓

```text
V2 100%
```

Benefits:

* Low risk
* Easy rollback
* Detect issues early

---

# Strategy 5: Blue-Green Deployment

Maintain two production environments.

```text
Blue

Current
```

```text
Green

New
```

Switch traffic only after validation.

```text
Users

↓

Blue

↓

Green
```

If problems occur:

```text
Green

↓

Rollback

↓

Blue
```

Advantages:

* Instant rollback
* Minimal downtime

---

# Strategy 6: Shadow Deployment

The new model receives the same requests but does not affect user responses.

```text
Client

↓

Production Model

↓

Response
```

At the same time:

```text
Same Request

↓

New Model

↓

Prediction Logged
```

Compare offline:

* Accuracy
* Latency
* Prediction differences

Excellent for testing high-risk models.

---

# Strategy 7: A/B Testing

Split users between models.

```text
Users

50%

↓

Model A
```

```text
Users

50%

↓

Model B
```

Compare business metrics:

* Revenue
* Conversion
* User satisfaction
* Retention

---

# Strategy 8: Time-Based Versioning

Useful for scheduled retraining.

```
fraud_model_2026_07_25

fraud_model_2026_08_01

fraud_model_2026_08_08
```

Works well when models are retrained weekly or daily.

---

# Metadata to Store

Every version should include metadata such as:

```json
{
  "version": "2.3.0",
  "dataset": "customer_v15",
  "git_commit": "a8f91c",
  "framework": "scikit-learn",
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.92,
  "created_at": "2026-07-25",
  "author": "CI Pipeline"
}
```

---

# Model Registry Workflow

```text
Training

↓

Evaluate

↓

Register

↓

Approve

↓

Deploy

↓

Monitor

↓

Retire
```

Only approved models should reach production.

---

# MLflow Example

Register a model:

```python
import mlflow

with mlflow.start_run():

    mlflow.log_metric("accuracy", 0.95)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="fraud_detector"
    )
```

Promote a version:

```text
Version 7

↓

Stage = Production
```

Rollback:

```text
Version 6

↓

Stage = Production
```

---

# Real-World Example

Suppose an e-commerce recommendation model is retrained weekly.

| Version | Dataset  | CTR  | Stage      |
| ------- | -------- | ---- | ---------- |
| V1      | Jan Data | 6.5% | Archived   |
| V2      | Feb Data | 7.1% | Archived   |
| V3      | Mar Data | 7.6% | Production |
| V4      | Apr Data | 7.8% | Staging    |
| V5      | May Data | 8.0% | Testing    |

Before promoting **V4**:

1. Run offline evaluation.
2. Execute integration tests.
3. Deploy with a 10% canary.
4. Monitor latency, CTR, and error rates.
5. If healthy, promote to Production.
6. If metrics degrade, roll back to V3.

---

# Best Practices

* **Version everything**: model, code, data, features, dependencies, and configuration.
* **Use immutable artifacts**: never modify a published model version.
* **Maintain full lineage** from dataset → code → model → deployment.
* **Automate registration** through CI/CD rather than manual uploads.
* **Require approval** before production deployment for critical applications.
* **Monitor production performance** and keep rollback procedures tested.
* **Store reproducibility metadata**, including random seeds and environment details.

---

# Comparison of Versioning Strategies

| Strategy            | Best For                  | Advantages                         | Drawbacks                          |
| ------------------- | ------------------------- | ---------------------------------- | ---------------------------------- |
| Sequential          | Small projects            | Simple and easy to understand      | Limited deployment control         |
| Stage-Based         | Enterprise workflows      | Clear promotion lifecycle          | Requires governance                |
| Champion-Challenger | Continuous improvement    | Safe comparison against production | Extra infrastructure               |
| Canary              | Production rollouts       | Low-risk incremental deployment    | More operational complexity        |
| Blue-Green          | Zero-downtime deployments | Instant rollback                   | Doubles infrastructure temporarily |
| Shadow              | High-risk validation      | No user impact                     | Additional compute cost            |
| A/B Testing         | Product optimization      | Measures business impact           | Needs sufficient traffic           |
| Time-Based          | Scheduled retraining      | Simple automation                  | Doesn't capture semantic changes   |

For production AI systems, organizations often combine several strategies. For example, they may **register every model version in a registry, promote it through staging, validate it with shadow traffic, deploy it via a canary rollout, and retain the previous champion for immediate rollback**. This layered approach provides both operational safety and strong governance.
