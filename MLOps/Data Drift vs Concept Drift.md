# Data Drift vs Concept Drift

One of the most common senior ML interview questions is:

> **"What is the difference between data drift and concept drift?"**

Although both degrade model performance, they occur for **different reasons**.

| Aspect                | Data Drift                                    | Concept Drift                          |
| --------------------- | --------------------------------------------- | -------------------------------------- |
| What changes?         | Input feature distribution                    | Relationship between inputs and target |
| Changes in X?         | ✅ Yes                                         | Sometimes, but not necessarily         |
| Changes in Y?         | Not required                                  | Usually yes (relationship changes)     |
| Can labels detect it? | No, labels aren't required                    | Yes, labels are typically needed       |
| Detection difficulty  | Easier                                        | Harder                                 |
| Solution              | Retrain with new data or update preprocessing | Retrain or redesign the model          |

---

# Understanding the Difference

Machine learning learns a function:

```text
Y = f(X)
```

Where:

* **X** = Features (Age, Income, Transaction Amount)
* **Y** = Target (Fraud, Default, Churn)

---

## Data Drift

In **data drift**, **X changes**.

The model receives a different distribution of input data than it saw during training.

```text
Training

X -------> Model -------> Y
```

↓

```text
Production

Different X -------> Same Model -------> Y
```

The relationship between X and Y is assumed to remain the same.

---

### Example 1: Loan Approval

Training data:

```text
Income

₹40,000

₹60,000

₹80,000

₹1,00,000
```

Average income:

```text
₹70,000
```

Production:

```text
Income

₹15,000

₹20,000

₹25,000
```

Average income:

```text
₹20,000
```

The income distribution has shifted.

---

### Example 2: E-commerce

Training:

```text
Average Order Value

₹1,000
```

Production:

```text
Average Order Value

₹3,000
```

Customer purchasing behavior has changed.

---

### Visualization

Training:

```text
***************
************
*********
****
```

Production:

```text
****
*********
***************
********************
```

The distribution has moved.

---

## Detecting Data Drift

One common statistical test is the Kolmogorov-Smirnov (KS) test.

```python
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(train_feature, production_feature)

if p_value < 0.05:
    print("Data drift detected")
```

Other techniques include:

* Population Stability Index (PSI)
* Jensen-Shannon Divergence
* Wasserstein Distance
* KL Divergence (for suitable distributions)

---

# Concept Drift

Concept drift is more serious.

The **relationship between X and Y changes**.

```text
Old

Income

↓

Default
```

↓

```text
New

Income

↓

Different Default Behaviour
```

The inputs may look similar, but the mapping to the target has changed.

---

### Example: Fraud Detection

Training:

```text
Large Transaction

↓

High Fraud Risk
```

Fraudsters adapt.

Production:

```text
Large Transaction

↓

Low Fraud Risk

Small Transactions

↓

High Fraud Risk
```

The fraud patterns have changed.

---

### Example: Loan Default

Before recession:

```text
Income = ₹80,000

↓

Low Default Risk
```

During recession:

```text
Income = ₹80,000

↓

Higher Default Risk
```

The feature values remain similar.

The relationship has changed.

---

### Visualization

Training:

```text
Income

↓

Loan Approval
```

Production:

```text
Income

↓

Different Approval Behaviour
```

---

## Detecting Concept Drift

You usually need **true labels**.

Example:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, predictions)

print(accuracy)
```

Training:

```text
Accuracy

96%
```

↓

Production:

```text
Accuracy

84%
```

If feature distributions are stable but accuracy drops after labels become available, concept drift is a strong possibility.

---

# Side-by-Side Example

Imagine a spam classifier.

## Case 1: Data Drift

Training emails:

```text
Mostly English Emails
```

Production:

```text
Mostly Spanish Emails
```

Different input distribution.

The spam rules may still work.

---

## Case 2: Concept Drift

Training:

```text
"Win Prize"

↓

Spam
```

Spammers evolve.

Production:

```text
"Limited Offer"

↓

Spam
```

The meaning of features has changed.

---

# Timeline Example

```text
January

Train Model
```

↓

```text
March

Customers change

↓

Data Drift
```

↓

```text
June

Market changes

↓

Concept Drift
```

---

# Production Monitoring Pipeline

```text
                  Production Data
                        │
        ┌───────────────┼───────────────┐
        ▼                               ▼
 Feature Distribution            Model Predictions
        │                               │
        ▼                               ▼
 Data Drift Monitor          Accuracy Monitor
        │                               │
        ▼                               ▼
      Alert                     Concept Drift
```

---

# Real-World Example: Ride-Sharing Pricing

Suppose you build a fare prediction model.

Training data:

| Distance (km) | Time   | Fare |
| ------------: | ------ | ---: |
|             5 | Normal | ₹200 |
|            10 | Normal | ₹400 |
|            15 | Normal | ₹600 |

## Scenario 1: Data Drift

A festival causes longer trips.

| Distance (km) | Time   |   Fare |
| ------------: | ------ | -----: |
|            20 | Normal |   ₹800 |
|            25 | Normal | ₹1,000 |

The input distribution (trip distance) changes.

---

## Scenario 2: Concept Drift

A new surge pricing policy is introduced.

| Distance (km) | Time |   Fare |
| ------------: | ---- | -----: |
|             5 | Peak |   ₹500 |
|            10 | Peak | ₹1,000 |
|            15 | Peak | ₹1,500 |

The relationship between features and fare has changed.

---

# How Production Teams Handle Them

## Data Drift

```text
Detect

↓

Investigate Features

↓

Validate Data Pipeline

↓

Retrain if Necessary
```

Focus on:

* Data quality
* Feature engineering
* Preprocessing
* Recent training data

---

## Concept Drift

```text
Detect

↓

Collect New Labels

↓

Retrain

↓

Evaluate

↓

Canary Deployment

↓

Production
```

Concept drift often requires retraining because the model's learned relationships are no longer valid.

---

# Interview Answer (2-Minute Version)

> **Data drift** occurs when the **distribution of input features changes** between training and production. For example, a fraud model trained on transactions averaging ₹1,000 starts receiving transactions averaging ₹5,000. We can often detect this without labels using statistical tests like KS, PSI, or Jensen-Shannon divergence.
>
> **Concept drift** occurs when the **relationship between inputs and the target changes**. For example, fraudsters change their behavior, so patterns that once indicated fraud no longer do. Detecting concept drift typically requires labeled production data because we need to observe a decline in prediction quality.
>
> In production, teams monitor **feature distributions** to catch data drift and **model performance metrics** (accuracy, precision, recall, business KPIs) to detect concept drift. Both types of drift should trigger investigation, and concept drift often leads to retraining and a controlled rollout of a new model.
