# Monitoring Model Drift

**Model drift monitoring** is the process of continuously checking whether an ML model's performance has degraded after deployment.

A model that performs well today may perform poorly in production months later because:

* Customer behavior changes
* Market conditions change
* New products are introduced
* Seasonal patterns emerge
* Fraudsters adapt
* The underlying relationship between inputs and outputs changes

Without drift monitoring, your model can silently become less accurate while still serving predictions.

---

# Types of Drift

There are three major types of drift.

```text
                    Production Model

                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
   Data Drift      Concept Drift     Prediction Drift
```

---

# 1. Data Drift (Feature Drift)

The **input data distribution changes**.

Training:

```text
Age

20 25 30 35 40 45 50
```

Production:

```text
Age

55 60 65 70 75
```

The model is receiving data it rarely or never saw during training.

Example:

A loan approval model was trained before inflation.

Training data:

```text
Income

₹40,000

₹60,000

₹80,000
```

Production:

```text
Income

₹20,000

₹25,000

₹30,000
```

The distribution has shifted significantly.

---

# Detecting Data Drift

One common approach is comparing distributions.

```python
import numpy as np
from scipy.stats import ks_2samp

train = np.random.normal(100, 10, 1000)
prod = np.random.normal(120, 10, 1000)

stat, p = ks_2samp(train, prod)

print(stat, p)
```

If the p-value is very small (commonly below 0.05), it suggests the distributions differ significantly.

---

# 2. Concept Drift

The relationship between inputs and outputs changes.

Training:

```text
Income

↓

Loan Default
```

Production:

```text
Income

↓

Loan Default

(New relationship)
```

Example:

Before an economic recession:

```text
High Income

↓

Low Default Risk
```

After a recession:

```text
High Income

↓

Higher Default Risk
```

The input distribution might remain similar, but the mapping from features to labels has changed.

---

# Detecting Concept Drift

Once labels become available, compare production performance with historical performance.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, predictions)

print(accuracy)
```

Example:

```text
Training Accuracy

96%
```

↓

```text
Production Accuracy

87%
```

A sustained drop is a signal to investigate.

---

# 3. Prediction Drift

The model's prediction distribution changes.

Training:

```text
Approved

80%

Rejected

20%
```

Production:

```text
Approved

40%

Rejected

60%
```

Even before labels arrive, this may indicate an issue.

---

# Monitoring Pipeline

```text
            Production Requests
                    │
                    ▼
             ML Inference API
                    │
                    ▼
           Prediction + Features
                    │
                    ▼
          Monitoring Pipeline
        ┌───────────┼─────────────┐
        ▼           ▼             ▼
  Data Drift   Prediction Drift  Latency
        │
        ▼
    Alerting System
        │
        ▼
 Retrain if Necessary
```

---

# Example Drift Monitor

```python
import numpy as np

class DriftMonitor:

    def __init__(self):
        self.history = []

    def add_prediction(self, value):
        self.history.append(value)

    def mean_prediction(self):
        return np.mean(self.history)

monitor = DriftMonitor()

monitor.add_prediction(0.72)
monitor.add_prediction(0.75)
monitor.add_prediction(0.69)

print(monitor.mean_prediction())
```

In production, you would compute these statistics over rolling windows rather than storing every prediction indefinitely.

---

# Monitoring Individual Features

Instead of monitoring only the model output, monitor every important feature.

```text
Customer Age

Average = 35

↓

Average = 48

Alert
```

```text
Transaction Amount

Average = ₹850

↓

Average = ₹2,500

Alert
```

Feature-level monitoring often detects issues earlier than accuracy monitoring.

---

# Drift Dashboard

```text
Feature Drift

Age                    Normal

Income                 Warning

Transaction Amount     Critical


Model Accuracy

94%


Latency

120 ms


Prediction Distribution

Stable
```

---

# Automated Alerts

Example thresholds:

```python
if accuracy < 0.90:
    send_alert()

if latency > 500:
    send_alert()

if drift_score > 0.20:
    send_alert()
```

Alerts can be sent to Slack, email, PagerDuty, or another incident management system.

---

# Retraining Workflow

```text
Drift Detected

↓

Alert

↓

Data Collection

↓

Retrain Model

↓

Evaluate

↓

Register New Model

↓

Canary Deployment

↓

Production
```

Avoid retraining automatically without evaluation—always verify that the new model performs better before promotion.

---

# Monitoring Stack

| Layer          | What to Monitor                        | Common Tools                    |
| -------------- | -------------------------------------- | ------------------------------- |
| Infrastructure | CPU, GPU, memory, disk                 | Prometheus, Grafana             |
| API            | Latency, throughput, error rate        | Prometheus, OpenTelemetry       |
| Features       | Distribution, missing values, outliers | Evidently AI, WhyLabs, Arize AI |
| Predictions    | Class distribution, confidence         | Prometheus, custom dashboards   |
| Labels         | Accuracy, precision, recall, F1        | MLflow, custom evaluation jobs  |
| Business       | Conversion, fraud rate, revenue        | Grafana, BI dashboards          |

---

# Real-World Example: Credit Card Fraud Detection

A fraud detection model is trained on 2025 transaction data.

**Normal production:**

| Metric              |  Value |
| ------------------- | -----: |
| Average transaction | ₹1,250 |
| Fraud rate          |   1.1% |
| Accuracy            |    96% |

Three months later:

| Metric              |  Value |
| ------------------- | -----: |
| Average transaction | ₹2,400 |
| Fraud rate          |   3.8% |
| Accuracy            |    89% |

The monitoring system detects:

* Transaction amount distribution has shifted (**data drift**).
* Accuracy has dropped after labels arrive (**concept drift**).
* Fraud predictions have increased significantly (**prediction drift**).

The response is:

1. Raise an alert.
2. Collect recent labeled transactions.
3. Retrain and evaluate a new model.
4. Register the model in the model registry.
5. Deploy it using a **canary deployment**.
6. Continue monitoring before rolling it out to 100% of traffic.

---

# Best Practices

* Monitor **feature distributions**, not just overall accuracy.
* Compare production data against the **training baseline** and recent production windows.
* Distinguish between **data drift**, **concept drift**, and **prediction drift**.
* Combine **technical metrics** (latency, errors) with **ML metrics** (accuracy, drift scores) and **business KPIs**.
* Use alerts with well-defined thresholds to catch issues early.
* Retrain only after validating that a new model improves performance.
* Keep a versioned history of models, datasets, and drift reports to support auditing and rollback.
