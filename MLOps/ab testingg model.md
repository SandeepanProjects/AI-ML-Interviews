# A/B Testing for Machine Learning Models (Production AI Engineer Perspective)

A/B testing is the **safest way to compare two or more models using real production traffic** before rolling out a new model to everyone.

Instead of replacing your production model immediately, you expose a **small percentage of users** to the new model and compare business outcomes.

---

# Why Do We Need A/B Testing?

Suppose your production fraud detection model has:

```text
Accuracy = 95%
```

You train a new model with:

```text
Accuracy = 97%
```

Should you deploy it immediately?

**No.**

Offline accuracy does **not** guarantee better production performance.

The new model may:

* Increase latency
* Cost more to run
* Produce more false positives
* Hurt user experience
* Reduce business revenue

That's why we run A/B tests.

---

# Basic Idea

Instead of this:

```text
Users

↓

New Model
```

Use:

```text
                Users

                  │

        Traffic Router / Load Balancer

          │                     │

      90% Traffic           10% Traffic

          │                     │

      Model A             Model B

(Current Production)      (Candidate)
```

Both models receive real production traffic.

---

# Example: Recommendation System

Suppose an e-commerce site recommends products.

Current model:

```text
Model A

Click Through Rate = 8%

Revenue/User = ₹320
```

New model:

```text
Model B

Unknown
```

Traffic split:

```text
90%

↓

Model A
```

```text
10%

↓

Model B
```

After one week:

| Metric       | Model A | Model B |
| ------------ | ------- | ------- |
| CTR          | 8%      | 10%     |
| Revenue/User | ₹320    | ₹355    |
| Latency      | 120 ms  | 140 ms  |

Model B performs better overall, so you may promote it.

---

# Real Production Architecture

```text
                    Internet

                        │

                 API Gateway

                        │

               Traffic Splitter

             ┌──────────┴──────────┐

             │                     │

         Model A               Model B

             │                     │

             └──────────┬──────────┘

                        │

                  Metrics Store

                        │

             Grafana / MLflow

                        │

                 Business Dashboard
```

---

# Example: Fraud Detection

Current model:

```text
Model A
```

New model:

```text
Model B
```

Incoming transaction:

```text
User pays ₹2,000
```

Routing:

```text
Random Number

↓

0.15
```

Since

```text
0.15 < 0.20
```

Request goes to

```text
Model B
```

Otherwise

```text
Model A
```

---

# Simple Python Example

```python
import random

def route_request():

    if random.random() < 0.2:
        return "Model B"

    return "Model A"

for _ in range(10):
    print(route_request())
```

Output

```text
Model A

Model A

Model B

Model A

Model B
```

Approximately 20% of traffic reaches Model B.

---

# FastAPI Example

```python
from fastapi import FastAPI
import random

app = FastAPI()

@app.post("/predict")
def predict(data: dict):

    if random.random() < 0.1:
        model = "candidate_model"
    else:
        model = "production_model"

    prediction = f"Prediction from {model}"

    return {
        "model": model,
        "prediction": prediction
    }
```

In production, the request router or service mesh (rather than application code) often handles traffic splitting.

---

# Metrics to Compare

Never compare only accuracy.

A production dashboard commonly tracks:

| ML Metrics  | System Metrics | Business Metrics      |
| ----------- | -------------- | --------------------- |
| Precision   | Latency        | Revenue               |
| Recall      | CPU            | Conversion Rate       |
| F1 Score    | GPU Usage      | Customer Satisfaction |
| ROC-AUC     | Memory         | Click Through Rate    |
| Calibration | Error Rate     | Retention             |

---

# Example Dashboard

```text
              Model A

Accuracy        95%

Latency         120 ms

Revenue         ₹5,20,000/day

CTR             7.5%
```

```text
              Model B

Accuracy        96%

Latency         140 ms

Revenue         ₹5,90,000/day

CTR             8.8%
```

Model B has slightly higher latency but significantly improves business results.

---

# How Long Should an A/B Test Run?

Running it for one hour is usually not enough.

Common practice:

* Collect enough users or requests for statistical confidence.
* Include weekdays/weekends if user behavior varies.
* Stop early only if there's a severe issue.

Typical duration ranges from several days to a few weeks depending on traffic volume.

---

# Statistical Significance

Suppose:

```text
Model A CTR = 8.0%

Model B CTR = 8.2%
```

Is Model B actually better?

Maybe.

Maybe not.

The difference could simply be random variation.

Teams use statistical tests (such as a two-proportion z-test or Bayesian methods) to determine whether the observed improvement is likely to be real before making a deployment decision.

---

# Shadow Testing vs A/B Testing

Many people confuse these.

## Shadow Testing

```text
User

↓

Production Model

↓

Response Returned
```

At the same time:

```text
Same Request

↓

Candidate Model

↓

Prediction Logged

↓

NOT Returned
```

The user never sees the candidate model's output.

Purpose:

* Compare predictions
* Measure latency
* Detect bugs
* Ensure stability

No business impact.

---

## A/B Testing

```text
User

↓

Traffic Split

↓

Model A OR Model B

↓

Prediction Returned
```

The user actually receives one model's prediction.

Business impact is measured directly.

---

# A/B Testing vs Canary Deployment

These are also different.

## A/B Testing

Goal:

```text
Which model is better?
```

Traffic:

```text
Model A → 50%

Model B → 50%
```

Decision is based on performance metrics.

---

## Canary Deployment

Goal:

```text
Can the new model run safely?
```

Traffic progression:

```text
1%

↓

5%

↓

10%

↓

25%

↓

50%

↓

100%
```

Focus is on operational safety, not comparison.

---

# Rollback Strategy

Suppose:

```text
Model B

↓

Latency doubles

↓

Revenue drops

↓

Prediction errors increase
```

Immediately switch traffic back:

```text
100%

↓

Model A
```

Because the original model remains deployed, rollback is fast.

---

# Enterprise Workflow

```text
Train New Model

        │

Offline Evaluation

        │

MLflow Model Registry

        │

Deploy Candidate

        │

A/B Testing

        │

Collect Metrics

        │

Statistical Analysis

        │

Winner Selected

        │

100% Production
```

---

# A/B Testing with MLflow

MLflow can log experiment details during the test:

```python
import mlflow

with mlflow.start_run():

    mlflow.log_param("experiment", "recommendation_ab_test")
    mlflow.log_param("traffic_split", "90_10")

    mlflow.log_metric("ctr", 0.102)
    mlflow.log_metric("latency_ms", 135)
    mlflow.log_metric("revenue_per_user", 355)
```

This provides a permanent record of the experiment and its outcomes.

---

# A/B Testing in Kubernetes

A common architecture is:

```text
                   Users

                      │

                Istio / Linkerd

                      │

         ┌────────────┴────────────┐

         │                         │

     Model A                  Model B

      90%                        10%

         │                         │

         └────────────┬────────────┘

                      │

               Prometheus

                      │

                 Grafana

                      │

                 MLflow Logs
```

Traffic routing is handled by the service mesh, while monitoring tools collect latency, error rates, and business metrics.

---

# Real-World Examples

### Recommendation System

Compare:

* Current recommender
* Transformer-based recommender

Metrics:

* Click-through rate
* Revenue per user
* Average order value

---

### Fraud Detection

Compare:

* XGBoost
* LightGBM

Metrics:

* Fraud detection rate
* False positives
* Investigation cost

---

### LLM Chatbot

Compare:

* GPT-4.1
* Fine-tuned Llama

Metrics:

* User satisfaction
* Response latency
* Hallucination rate
* Cost per request
* Task completion rate

---

# Interview Takeaway

If asked **"How do you safely deploy a new ML model?"**, a strong answer is:

> First, validate the model offline using a holdout test set. Next, deploy it as a candidate alongside the current production model. Start with shadow testing if possible, then run an A/B test by sending a controlled percentage of production traffic to the new model. Monitor ML metrics, system metrics, and business KPIs, and verify that improvements are statistically significant. If the new model consistently outperforms the existing one without operational issues, gradually increase traffic or complete the rollout. If performance degrades, immediately route traffic back to the existing production model.
