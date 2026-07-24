# Model Drift Detection (Production Guide)

Model drift detection is one of the most important topics in MLOps and AI production systems.

A machine learning model performs well when the **training data distribution** is similar to the **production data distribution**.

Over time, the real world changes:

* Customer behavior changes
* Market conditions change
* Fraud patterns evolve
* Language evolves
* User queries change

When this happens, model performance degrades. This is called **model drift**.

---

# What is Model Drift?

Suppose we trained a loan approval model.

Training Data (2024)

| Age | Salary | Approved |
| --- | ------ | -------- |
| 25  | 40K    | Yes      |
| 35  | 60K    | Yes      |
| 55  | 30K    | No       |

Model accuracy

```text
95%
```

One year later:

Production Data

| Age | Salary | Approved |
| --- | ------ | -------- |
| 22  | 120K   | Yes      |
| 27  | 150K   | Yes      |
| 31  | 180K   | Yes      |

The salary distribution has changed dramatically.

The model still predicts using old patterns.

Accuracy drops.

This is drift.

---

# Types of Drift

```text
                  Model Drift
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
Data Drift     Concept Drift    Prediction Drift
```

---

# 1. Data Drift (Covariate Drift)

The **input features change**.

Training

```text
Salary

20K

30K

40K

50K
```

Production

```text
Salary

100K

120K

180K

250K
```

The distribution changed.

But the relationship between input and output may still be the same.

---

Example

Training

```python
Age

20-40
```

Production

```python
Age

55-80
```

Different distribution.

---

# 2. Concept Drift

The relationship between input and output changes.

Example

Old fraud model

```text
Large transactions

↓

Fraud
```

Today

Fraudsters learn new tricks.

Now

```text
Small transactions

↓

Fraud
```

Inputs look similar.

Labels changed.

Model becomes inaccurate.

---

# 3. Prediction Drift

The predictions themselves change.

Training

```text
Positive

52%

Negative

48%
```

Production

```text
Positive

96%

Negative

4%
```

Prediction distribution shifted.

Usually indicates a problem.

---

# Real Production Pipeline

```text
                Training
                   │
                   ▼
              Train Model
                   │
                   ▼
          Deploy to Production
                   │
                   ▼
             Live Predictions
                   │
                   ▼
          Drift Detection System
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
 No Drift                Drift Detected
        │                      │
        ▼                      ▼
 Continue             Retrain Model
```

---

# Detecting Data Drift

Suppose

Training

```python
import numpy as np

train = np.random.normal(
    50,
    10,
    1000
)
```

Production

```python
production = np.random.normal(
    70,
    10,
    1000
)
```

Clearly shifted.

---

## Method 1: Kolmogorov-Smirnov (KS) Test

The KS test compares two distributions.

```python
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(
    train,
    production
)

print(statistic)
print(p_value)
```

Output

```text
Statistic = 0.42

P-value = 0.00001
```

Decision

```python
if p_value < 0.05:
    print("Data Drift Detected")
```

---

# Method 2: Population Stability Index (PSI)

One of the most common methods in finance.

Formula

```text
PSI

=

Σ

(Expected−Actual)

×

ln(Expected/Actual)
```

Interpretation

| PSI      | Meaning           |
| -------- | ----------------- |
| <0.1     | Stable            |
| 0.1–0.25 | Moderate Drift    |
| >0.25    | Significant Drift |

---

Simple implementation

```python
import numpy as np

def calculate_psi(expected, actual, bins=10):

    breakpoints = np.percentile(
        expected,
        np.arange(0, 101, 100 / bins)
    )

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_perc = (
        expected_counts /
        len(expected)
    ) + 1e-6

    actual_perc = (
        actual_counts /
        len(actual)
    ) + 1e-6

    psi = np.sum(
        (expected_perc - actual_perc) *
        np.log(expected_perc / actual_perc)
    )

    return psi
```

Usage

```python
psi = calculate_psi(train, production)

print(psi)
```

---

# Detecting Prediction Drift

Training predictions

```python
train_predictions = [
    0.2,
    0.3,
    0.4,
    0.6
]
```

Production

```python
live_predictions = [
    0.95,
    0.97,
    0.98
]
```

Compare distributions

```python
ks_2samp(
    train_predictions,
    live_predictions
)
```

---

# Detecting Concept Drift

Concept drift requires **ground truth labels**.

Example

Training

```text
Accuracy

95%
```

Production

```text
Accuracy

74%
```

Performance degraded.

Monitor metrics

```python
accuracy

precision

recall

f1

auc

log_loss
```

If accuracy keeps falling,

trigger retraining.

---

# Drift Monitoring Service

```python
from scipy.stats import ks_2samp

class DriftDetector:

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect(
        self,
        training,
        production
    ):

        statistic, p = ks_2samp(
            training,
            production
        )

        return {

            "drift": p < self.threshold,

            "p_value": p,

            "statistic": statistic

        }
```

Usage

```python
detector = DriftDetector()

result = detector.detect(
    train,
    production
)

print(result)
```

---

# Production Monitoring

Run every hour.

```python
while True:

    live_data = load_recent_data()

    drift = detector.detect(
        train_distribution,
        live_data
    )

    if drift["drift"]:

        send_slack_alert()

    time.sleep(3600)
```

---

# Production Architecture

```text
                    Production
                         │
                         ▼
                  Feature Store
                         │
                         ▼
                 Live Predictions
                         │
                         ▼
                Drift Detection
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
      Data Drift    Prediction Drift   Concept Drift
         │               │                │
         └───────────────┼────────────────┘
                         ▼
                    Alert System
                         │
                         ▼
                 Retraining Pipeline
                         │
                         ▼
                  Model Registry
                         │
                         ▼
                  Production Model
```

---

# Using Evidently AI

```python
from evidently.report import Report
from evidently.presets import DataDriftPreset

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=train_df,
    current_data=production_df
)

report.save_html("drift.html")
```

Evidently automatically checks:

* Feature drift
* Target drift
* Dataset drift
* Missing values
* Feature distributions
* Statistical tests

---

# Using WhyLabs

```python
from whylogs import log

log(train_df)

log(production_df)
```

WhyLabs provides:

* Drift dashboards
* Alerts
* Monitoring
* Explainability

---

# Production Folder Structure

```text
mlops/
│
├── drift_detector.py
├── psi.py
├── ks_test.py
├── monitoring.py
├── alert.py
├── retraining.py
├── scheduler.py
├── metrics.py
├── dashboards/
└── tests/
```

---

# Production Metrics Dashboard

Monitor:

### Data

* Feature drift
* Missing values
* Null rate
* Outliers

### Predictions

* Prediction distribution
* Confidence distribution
* Class imbalance

### Model

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Business

* Revenue
* Conversion rate
* Fraud rate
* Customer satisfaction

---

# Production Retraining Pipeline

```text
Live Data
    │
    ▼
Drift Detection
    │
    ▼
Drift Threshold Exceeded
    │
    ▼
Collect New Labeled Data
    │
    ▼
Train New Model
    │
    ▼
Evaluate
    │
    ▼
Register Model
    │
    ▼
Canary Deployment
    │
    ▼
Production
```

---

# Best Practices Used by Senior AI Engineers

1. **Monitor feature distributions** continuously using KS tests, PSI, or Jensen-Shannon divergence.
2. **Track model performance** (accuracy, precision, recall, F1, AUC) whenever ground-truth labels become available.
3. **Monitor prediction distributions** to detect unexpected shifts even before labels arrive.
4. **Set alert thresholds** (for example, PSI > 0.25 or statistically significant KS-test results) and integrate alerts with Slack, PagerDuty, or email.
5. **Use dashboards** (Prometheus + Grafana, Evidently AI, Arize Phoenix, WhyLabs) to visualize drift trends over time.
6. **Automate retraining**, but only after validating that a newly trained model outperforms the current production model.
7. **Deploy safely** using canary or shadow deployments before rolling out the new model to all traffic.

---

# Senior AI Engineer Interview Questions

### Why isn't accuracy enough to detect drift?

Accuracy requires labeled production data, which may arrive days or weeks later. Data drift detection methods (such as KS tests or PSI) work immediately on incoming features, allowing earlier detection.

---

### What's the difference between data drift and concept drift?

| Data Drift                     | Concept Drift                                    |
| ------------------------------ | ------------------------------------------------ |
| Feature distribution changes   | Relationship between features and labels changes |
| Can be detected without labels | Usually requires labeled data                    |
| Example: customer ages shift   | Example: fraud patterns evolve                   |

---

### When should you retrain a model?

Retraining should be triggered by a combination of signals:

* Significant feature drift
* Sustained degradation in model performance
* Business KPI decline (e.g., conversion or fraud detection)
* Sufficient new labeled data

Avoid retraining solely because drift was detected—first verify that a new model actually improves performance before deployment.
