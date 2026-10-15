# Precision vs Recall vs F1-Score in Machine Learning

In a production ML project, precision, recall, and F1-score help you understand what your model gets right, what it misses, and what kind of errors it makes.

Let's use a financial fraud detection system as our running example. This is a useful way to understand why accuracy alone is often insufficient for imbalanced datasets.

## 1. Understand the four possible outcomes first

Imagine your model examines 1,000 transactions and predicts whether each transaction is fraudulent.

- Positive: Fraudulent transaction.
- Negative: Legitimate transaction.

The model can make four kinds of predictions:

True Positive (TP)

# 80

Fraud correctly detected

False Positive (FP)

# 20

Legitimate transactions incorrectly flagged

False Negative (FN)

# 40

Fraud that the model missed

True Negative (TN)

# 860

Legitimate transactions correctly accepted

Illustrative example: 120 transactions were flagged as fraud; 100 were actually fraudulent.

These four values form the confusion matrix, which is the foundation of precision and recall.

| Actual / Predicted | Predicted fraud | Predicted legitimate |
| ------------------ | --------------- | -------------------- |
| Actual fraud       | TP = 80         | FN = 40              |
| Actual legitimate  | FP = 20         | TN = 860             |

## 2. Precision: When the model raises an alert, how often is it correct?

Precision answers:

Of all the transactions the model flagged as fraud, how many really were fraud?

\\[ \text{Precision}=\frac{TP}{TP+FP} \\]

Using our example:

\\[ \text{Precision}=\frac{80}{80+20}=0.80 \\]

## Precision = 80%

4 out of 5

Of every 100 fraud alerts, approximately 80 are correct.

### Why does precision matter in production?

Suppose a fraud detection system sends alerts to a team of human investigators.

If precision is low:

- Investigators receive many false alarms.
- Operational costs increase.
- Legitimate customers may experience unnecessary account restrictions.
- The team may struggle to prioritize genuine fraud.

High precision is especially valuable when a false positive is expensive.

Example use cases: fraud alerts, spam filtering when legitimate emails must not be blocked, and compliance review systems where false alarms consume significant reviewer time.

## 3. Recall: Of all actual fraud cases, how many did the model detect?

Recall answers:

Of all transactions that really were fraudulent, how many did the model successfully identify?

\\[ \text{Recall}=\frac{TP}{TP+FN} \\]

Using our example:

\\[ \text{Recall}=\frac{80}{80+40}=0.667 \\]

## Recall = 66.7%

2 out of 3

Of every 100 actual fraud cases, approximately 67 are detected and 33 are missed.

### Why does recall matter in production?

Low recall means the system misses too many real positive cases.

In fraud detection, that may mean financial losses. In medical screening, it may mean missing people who need further assessment. In compliance, it may mean failing to flag potentially prohibited claims.

High recall is especially important when false negatives are expensive.

A model with high recall can still have poor precision, however, because it may flag many legitimate cases too.

## 4. F1-score: A balance between precision and recall

F1-score combines precision and recall using their harmonic mean.

\\[ F_1=2\cdot\frac{P\cdot R}{P+R} \\]

Here, \\(P\\) means precision and \\(R\\) means recall.

For our example:

\\[ F_1=2\cdot\frac{0.80\cdot0.667}{0.80+0.667} \approx 0.727 \\]

## F1-score = 72.7%

A single summary metric combining precision and recall.

F1-score is useful when you need to balance missed positives against false alarms and want one metric to compare models.

But remember:

- A high F1-score does not guarantee high precision and high recall individually.
- F1 ignores true negatives.
- F1 treats precision and recall symmetrically.
- F1 may be unsuitable if the business costs of false positives and false negatives are very different.

If missing fraud is much more costly than investigating a false alarm, a recall-focused metric or an explicit business cost function may be more appropriate.

## 5. Calculate precision, recall and F1-score using Python

Let's calculate all three metrics with scikit-learn.

```
from sklearn.metrics import (    confusion_matrix,    precision_score,    recall_score,    f1_score,    classification_report,    accuracy_score)# Actual labels: 1 = fraud, 0 = legitimatey_true = [1] * 120 + [0] * 880# Predictions: 80 fraud cases detected, 40 missed,# 20 legitimate transactions falsely flagged, 860 correcty_pred = (    [1] * 80 + [0] * 40 +    [1] * 20 + [0] * 860)print("Confusion matrix:")print(confusion_matrix(y_true, y_pred))print("Accuracy:", accuracy_score(y_true, y_pred))print("Precision:", precision_score(y_true, y_pred))print("Recall:", recall_score(y_true, y_pred))print("F1-score:", f1_score(y_true, y_pred))print("\nClassification report:")print(classification_report(    y_true,    y_pred,    target_names=["Legitimate", "Fraud"],    zero_division=0))
```

Expected results, approximately:

```
Accuracy: 0.94
Precision: 0.80
Recall: 0.667
F1-score: 0.727
```

The accuracy is 94%, but recall is only 66.7%. The model misses one-third of all fraudulent transactions.

This illustrates why production teams should not select a model based on accuracy alone.

### What does `classification_report()` tell you?

It reports precision, recall, F1-score and support for each class.

- Support: Number of actual examples belonging to a class.
- Macro average: Computes the metric for each class and averages them equally.
- Weighted average: Averages class metrics weighted by each class's support.
- Accuracy: Proportion of all predictions that are correct.

For fraud detection, always inspect the row for the fraud class, not just the weighted average.

## 6. Train a real ML model and evaluate the metrics

Now let's connect these metrics to a complete model-training workflow.

We will create an imbalanced dataset, train a logistic regression classifier, and evaluate its predictions.

```
from sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import (    classification_report,    confusion_matrix,    average_precision_score)# 1. Create an imbalanced datasetX, y = make_classification(    n_samples=10_000,    n_features=10,    n_informative=5,    n_redundant=2,    weights=[0.95, 0.05],    random_state=42)# 2. Split before fitting any preprocessingX_train, X_test, y_train, y_test = train_test_split(    X, y,    test_size=0.20,    stratify=y,    random_state=42)# 3. Build a reproducible preprocessing + model pipelinemodel = Pipeline([    ("scaler", StandardScaler()),    ("classifier", LogisticRegression(        max_iter=1000,        class_weight=None    ))])# 4. Train on the training datamodel.fit(X_train, y_train)# 5. Predict on the untouched test datay_pred = model.predict(X_test)y_prob = model.predict_proba(X_test)[:, 1]# 6. Evaluateprint(confusion_matrix(y_test, y_pred))print(classification_report(    y_test, y_pred, zero_division=0))print(    "Average precision:",    average_precision_score(y_test, y_prob))
```

The exact metrics depend on the generated dataset and split.

Production lesson: use the same held-out test set to compare candidate models fairly. For a real time-dependent financial dataset, a chronological split may be more appropriate than a random split.

## 7. The most important production concept: the prediction threshold

A classifier often produces a probability-like score before converting it into a class.

For example:

```
Transaction A -> fraud score 0.92
Transaction B -> fraud score 0.65
Transaction C -> fraud score 0.28
Transaction D -> fraud score 0.07
```

A typical binary classifier uses a threshold of `0.5` by default:

- Score greater than or equal to 0.5 → predict fraud.
- Score below 0.5 → predict legitimate.

But 0.5 is not automatically the best production threshold.

Lowering the threshold generally catches more positive cases, increasing recall, but it often also increases false positives and reduces precision. Raising it generally has the opposite effect. The exact results depend on the model and dataset.

Here is how to evaluate multiple thresholds using validation data.

```
from sklearn.metrics import precision_score, recall_score, f1_score# Train/validation/test separation is recommended for production.# Here, X_val and y_val represent a validation set that was# separated from the training data before model selection.model.fit(X_train, y_train)val_prob = model.predict_proba(X_val)[:, 1]for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:    val_pred = (val_prob >= threshold).astype(int)    precision = precision_score(        y_val, val_pred, zero_division=0    )    recall = recall_score(        y_val, val_pred, zero_division=0    )    f1 = f1_score(        y_val, val_pred, zero_division=0    )    print(        f"Threshold={threshold:.2f} | "        f"Precision={precision:.3f} | "        f"Recall={recall:.3f} | "        f"F1={f1:.3f}"    )
```

The snippet assumes you have already created `X_val` and `y_val` by splitting the original training portion into training and validation sets. Do not use the final test set to select the threshold.

This is a critical production workflow:

1. Train the model on training data.
2. Choose a threshold using validation data and a business objective.
3. Freeze the model and threshold.
4. Evaluate once on the untouched test set.
5. Monitor the same metrics after deployment.

Here is the missing validation split to make the threshold-tuning example runnable end to end. Use it before fitting the model:

```
from sklearn.model_selection import train_test_split# First split: hold out the final test set.X_dev, X_test, y_dev, y_test = train_test_split(    X, y,    test_size=0.20,    stratify=y,    random_state=42)# Second split: create training and validation sets.X_train, X_val, y_train, y_val = train_test_split(    X_dev, y_dev,    test_size=0.25,    stratify=y_dev,    random_state=42)# This gives approximately:# 60% training, 20% validation, 20% testing.
```

Use `X_train` and `y_train` to fit the model, `X_val` and `y_val` to choose the threshold, and `X_test` and `y_test` only for final evaluation.

## 8. How senior ML engineers choose these metrics in production

Different products have different error costs. A production team should agree on the business objective before choosing its evaluation metric.

| Production use case         | Typical priority                                        | Reason                                                                      |
| --------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| Fraud detection             | Recall and precision together                           | Catch fraud without overwhelming investigators                              |
| Medical screening           | High recall                                             | Minimize missed cases requiring further assessment                          |
| Spam filtering              | Often high precision                                    | Avoid incorrectly blocking legitimate messages                              |
| Financial compliance review | Recall for material violations, with precision controls | Avoid missing serious violations while controlling review workload          |
| Recommendation systems      | Depends on ranking and business goals                   | Precision\@K, Recall\@K, NDCG or other ranking metrics may be more relevant |

These are starting points, not universal rules. For example, a fraud system that automatically blocks payments may prioritize precision more heavily than one that simply sends transactions for review.

### A realistic production architecture

1\. Data pipeline

Validated features and ground-truth labels

2\. Model training and evaluation

Baseline, cross-validation, precision, recall, F1 and PR-AUC

3\. Threshold selection

Choose based on business costs and review capacity

4\. Model registry and deployment

Version model, preprocessing, threshold and evaluation results

5\. Production monitoring

Measure delayed ground-truth outcomes, drift, false alarms and missed cases

In a real production system, you also need to account for delayed labels. A fraud label may only become reliable after an investigation, so yesterday's predictions cannot necessarily be evaluated today.

Monitor precision, recall and F1 when trustworthy labels become available. Track prediction volume, score distributions, data drift, latency and operational costs in the meantime. If the model or data distribution changes, investigate and revalidate before changing the threshold.

## 9. Interview-ready answer

What is the difference between precision, recall and F1-score?

- Precision measures how many predicted positive cases are actually positive.
- Recall measures how many actual positive cases the model detects.
- F1-score is the harmonic mean of precision and recall.

In production, I choose metrics based on the relative costs of false positives and false negatives. I evaluate on representative validation and test data, tune the classification threshold using validation data, and monitor metrics using reliable production labels. For imbalanced classification, I also examine the confusion matrix and average precision rather than relying only on accuracy.

The key takeaway: Precision tells you how trustworthy the alerts are; recall tells you how many real cases you catch; F1 summarizes their balance. None should be optimized blindly without considering the business impact of errors.