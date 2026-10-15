# How to Handle Imbalanced Datasets in Machine Learning — With Proper Python Code

Handling imbalanced datasets is an important skill for a senior ML engineer because real-world datasets are often imbalanced. Examples include fraud detection, financial compliance, customer churn, medical diagnosis, and suspicious transaction detection.

The key principle is:

Do not try to make the dataset balanced blindly. First understand the business problem, choose the right evaluation metrics, and then select an appropriate strategy.

We’ll go from fundamentals to a practical, leakage-safe ML pipeline using Python and scikit-learn.

## 1. What is an imbalanced dataset?

An imbalanced dataset is one where one class has significantly more examples than another.

Imagine a financial transaction fraud-detection system.

## Example: 100,000 transactions

Legitimate transactions

### 99,000 (99%)

Fraudulent transactions

### 1,000 (1%)

Class 0 = legitimate · Class 1 = fraud

Here, the legitimate class is the majority class, and fraud is the minority class.

### Why is this a problem?

Suppose a model predicts every transaction as legitimate.

- Accuracy = 99%.
- Fraud detection recall = 0%.
- Every fraudulent transaction is missed.

The model has excellent accuracy but is useless for detecting fraud.

This is why accuracy alone can be misleading for imbalanced classification.

## 2. Understand the important evaluation metrics

Before balancing a dataset, understand what you want the model to optimize.

| Metric                     | Meaning                                                                    | When it matters                        |
| -------------------------- | -------------------------------------------------------------------------- | -------------------------------------- |
| Precision                  | Of predicted positives, how many were actually positive?                   | Avoiding false fraud alerts            |
| Recall                     | Of actual positives, how many did the model detect?                        | Catching more fraud cases              |
| F1-score                   | Harmonic mean of precision and recall                                      | Balancing precision and recall         |
| PR-AUC / Average Precision | Measures ranking quality for the positive class using precision and recall | Often useful for rare positive classes |
| ROC-AUC                    | Measures how well scores rank positives above negatives                    | Overall discrimination                 |
| Balanced accuracy          | Average recall across classes                                              | Evaluating both classes more fairly    |

The underlying confusion matrix contains four outcomes:

True Positive (TP)

Fraud correctly detected.

False Positive (FP)

Legitimate transaction incorrectly flagged.

False Negative (FN)

Fraud missed by the model.

True Negative (TN)

Legitimate transaction correctly accepted.

The formulas are:

\\[ \text{Precision}=\frac{TP}{TP+FP} \\]

\\[ \text{Recall}=\frac{TP}{TP+FN} \\]

\\[ F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall} \\]

For fraud detection, missing fraud can be very costly, so recall may be important. But if investigators have limited capacity, precision and the number of false alerts matter too.

There is no universally best metric. Select it based on business costs and operational constraints.

## 3. The main techniques for handling imbalanced datasets

[10 Scikit-learn Tricks for Working with Imbalanced Datasets | by Hash Block | Medium](https://images.openai.com/static-rsc-4/BtVePAERX9EcL0FAkgs2rfLxcv9OPK3sAAmMsRQOxU10aEjPiyMgaZJic2P3rCNuQYa-1ZQLdABTwW5AJi9B3msb0QIfPpLENlmVXN8v5md-FmluOt6KUcRlYfZtFdZVsoFVrby2CvREHA-MBjvNfct3W39K1T9cBx5oVeEzJYY?purpose=inline)

1\. Class weighting

Penalize mistakes on the minority class more heavily. This is often a good baseline because it does not create synthetic examples or discard training records.

[Toxic Comments Classification — NLP Project | by John Feng | Medium](https://images.openai.com/static-rsc-4/D2rIux2sgLulHNg2TKq4cuUgUiJcOuIXLtNtwDTsJjo-w1nYaAsB3kRnpNRk6Cs3LE8PcTyzkLd2QBDGB5KvQ5RbIzZvx5R4c1EHpUw4PL5phMQvJFuavVR3nLsOYzcomG9epUCLkTrNwnv-h8cNyWwxeBBdF40X3vStlQy-KYc?purpose=inline)

2\. Random oversampling

Duplicate minority-class examples in the training data to give them greater influence.

[Sample generator used in SMOTE-like samplers — Version 0.15.dev0](https://images.openai.com/static-rsc-4/iT1lRc7Im6200x2JlBpG59JWB4Wf7hZUqeXPbq0_XERSTnLFGBLJHA1WtuXNPBZlA56w1q4xSSo-GbcH8Y37dXKepqzRpHEaHn2DWkOeDjIOAsJW2Jcb7tlDdSbAUsyk0IrPVhg4SWwLk45excsonpqql1xughEqz7wfcPbfXhA?purpose=inline)

3\. SMOTE

Create synthetic minority-class samples by interpolating between nearby minority examples. This can help, but synthetic records may be unrealistic for some datasets.

[Medium](https://images.openai.com/static-rsc-4/AKK63dBixCIKQQcGeK3UfXgu8NtJpr3cZ_d-gOm4fcL8tkpE5IOSLJZ3INGZnzRUCLM4J6CguHi2yN93M1k33ytpYMIqyg0MniAjRGN5NYn4YElNUjLhZmMSD6XStBsD4Cw-uvCL6zZv90Va5UmJeW2K6qogdASJlQQbt89GWgk?purpose=inline)

4\. Random undersampling

Remove some majority-class examples. This reduces training size but can discard valuable information.

[Loan Default Prediction — A Deep Dive into Machine Learning and Class Imbalance | by Dardeep Somel | Medium](https://images.openai.com/static-rsc-4/wnLQ4NKOU_ensy-54TPpdJpsFeAErs0Z8ENE9XpUprGpGpmvJ3PMfikAcnT9EUYXE6k1riSxh23DsCh9IOV6dgTzV_AZoxM-7ttqYHFOQOiFYLlDSqt8XWLUZQspaasRhFVOl3SA-ufLuJ_VASqjvptjecsdAlMmgOyGrzQZvr4?purpose=inline)

5\. Decision-threshold tuning

Keep the trained model but adjust the probability threshold used to classify a record as positive. This is distinct from resampling the dataset.

Other options include collecting more minority-class examples, improving features, using anomaly-detection methods when labels are scarce, and choosing models designed for imbalanced classification.

## 4. Build a complete imbalanced-classification example

We'll build a fraud-detection model using synthetic transaction-like data. The example demonstrates the workflow, but the generated data is not real fraud data and should not be treated as a production-ready fraud detector.

### Step 1: Install the libraries

```
pip install numpy pandas scikit-learn imbalanced-learn
```

We will use:

- `scikit-learn` for splitting, model training, and evaluation.
- `imbalanced-learn` for SMOTE and sampling pipelines.
- `numpy` and `pandas` for data manipulation.

### Step 2: Create a highly imbalanced dataset

```
import numpy as npimport pandas as pdfrom sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitX, y = make_classification(    n_samples=20_000,    n_features=20,    n_informative=8,    n_redundant=4,    n_classes=2,    weights=[0.99, 0.01],    flip_y=0,    class_sep=1.0,    random_state=42,)X = pd.DataFrame(    X,    columns=[f"feature_{i}" for i in range(X.shape[1])])y = pd.Series(y, name="is_fraud")print("Class counts:")print(y.value_counts())print("\nClass percentages:")print((y.value_counts(normalize=True) * 100).round(2))
```

Explanation:

- `n_samples=20_000`: creates 20,000 observations.
- `n_features=20`: each observation has 20 features.
- `weights=[0.99, 0.01]`: approximately 99% belong to class 0 and 1% to class 1.
- `flip_y=0`: avoids deliberately flipping labels in this synthetic example.
- `random_state=42`: makes the example reproducible.

### Step 3: Split the dataset correctly

Keep training, validation, and test data separate. Use stratification so each split retains approximately the same class ratio.

```
# 60% train, 40% temporaryX_train, X_temp, y_train, y_temp = train_test_split(    X,    y,    test_size=0.40,    stratify=y,    random_state=42,)# Divide the remaining 40% equally:# 20% validation, 20% testX_val, X_test, y_val, y_test = train_test_split(    X_temp,    y_temp,    test_size=0.50,    stratify=y_temp,    random_state=42,)for name, labels in [    ("Train", y_train),    ("Validation", y_val),    ("Test", y_test),]:    print(        name,        "samples =", len(labels),        "positive rate =", round(labels.mean(), 4),        "class counts =", labels.value_counts().to_dict(),    )
```

Why three splits?

| Split      | Purpose                                        |
| ---------- | ---------------------------------------------- |
| Training   | Fit the model and any sampling strategy        |
| Validation | Compare models and tune the decision threshold |
| Test       | Estimate final performance on untouched data   |

Critical rule: Never apply SMOTE or oversampling before splitting. Otherwise, related or synthetic examples can leak into validation or test data, making performance look better than it really is.

### Step 4: Build a reusable evaluation function

```
from sklearn.metrics import (    accuracy_score,    average_precision_score,    balanced_accuracy_score,    classification_report,    confusion_matrix,    f1_score,    precision_score,    recall_score,    roc_auc_score,)def evaluate_model(    model,    X_eval,    y_eval,    threshold=0.5,    title="Model",):    probabilities = model.predict_proba(X_eval)[:, 1]    predictions = (probabilities >= threshold).astype(int)    print(f"\n--- {title} ---")    print(f"Threshold:          {threshold:.3f}")    print(f"Accuracy:           {accuracy_score(y_eval, predictions):.4f}")    print(f"Balanced accuracy:  {balanced_accuracy_score(y_eval, predictions):.4f}")    print(f"Precision:          {precision_score(y_eval, predictions, zero_division=0):.4f}")    print(f"Recall:             {recall_score(y_eval, predictions, zero_division=0):.4f}")    print(f"F1-score:           {f1_score(y_eval, predictions, zero_division=0):.4f}")    print(f"Average precision:  {average_precision_score(y_eval, probabilities):.4f}")    print(f"ROC-AUC:            {roc_auc_score(y_eval, probabilities):.4f}")    print("\nConfusion matrix:")    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))
```

A few important details:

- `predict_proba()` gives the model's estimated positive-class probability.
- `threshold=0.5` means class 1 is predicted when the probability is at least 0.5.
- Average precision summarizes precision-recall performance across thresholds.
- ROC-AUC measures ranking quality, but can look strong even when precision for a rare positive class is poor.
- The confusion matrix shows how many positive examples were missed and how many negative examples were incorrectly flagged.

### Step 5: Establish a baseline

First, train a model that predicts only the majority class.

```
from sklearn.dummy import DummyClassifierbaseline = DummyClassifier(strategy="most_frequent")baseline.fit(X_train, y_train)evaluate_model(    baseline,    X_val,    y_val,    title="Majority-class baseline",)
```

This is a useful sanity check. The model may report approximately 99% accuracy, yet have zero recall for fraud.

That illustrates why a high accuracy score does not necessarily mean a useful model.

## 5. Technique 1 — Class weighting

Class weighting tells the learning algorithm that minority-class errors should have a larger penalty.

For example, in scikit-learn:

```
from sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerweighted_model = Pipeline([    ("scaler", StandardScaler()),    (        "classifier",        LogisticRegression(            class_weight="balanced",            max_iter=2000,            random_state=42,        ),    ),])weighted_model.fit(X_train, y_train)weighted_val_probabilities, _ = evaluate_model(    weighted_model,    X_val,    y_val,    title="Class-weighted logistic regression",)
```

### How does `class_weight="balanced"` work?

Scikit-learn assigns each class a weight inversely proportional to its frequency:

\\[ w_c = \frac{N}{K \cdot N_c} \\]

Where:

- \\(N\\) = total number of training observations.
- \\(K\\) = number of classes.
- \\(N_c\\) = number of training observations in class \\(c\\).

For 99% legitimate and 1% fraud, the minority class receives a much larger weight than the majority class.

Advantages

- No synthetic records are created.
- No real observations need to be discarded.
- Simple to implement and tune.
- A strong baseline for many classification problems.

Limitations

- It may increase false positives.
- It does not guarantee good minority-class recall.
- It may change the calibration of predicted probabilities.

Class weighting is often the first technique I would test before introducing SMOTE.

## 6. Technique 2 — Random oversampling

Random oversampling duplicates minority-class examples in the training data.

Suppose the training data contains:

- 11,880 legitimate transactions
- 120 fraudulent transactions

An oversampling strategy can duplicate fraud records so that they have more influence during training.

```
from imblearn.over_sampling import RandomOverSamplerfrom imblearn.pipeline import Pipeline as ImbPipelinefrom sklearn.linear_model import LogisticRegressionfrom sklearn.preprocessing import StandardScaleroversampling_model = ImbPipeline([    ("scaler", StandardScaler()),    (        "sampler",        RandomOverSampler(            sampling_strategy=0.20,            random_state=42,        ),    ),    (        "classifier",        LogisticRegression(            max_iter=2000,            random_state=42,        ),    ),])oversampling_model.fit(X_train, y_train)evaluate_model(    oversampling_model,    X_val,    y_val,    title="Random oversampling",)
```

Here, `sampling_strategy=0.20` means the resampled minority-to-majority ratio is 0.20 for binary classification. It does not mean that the final minority class will make up 20% of the entire dataset.

Notice that scaling and sampling are inside an imbalanced-learn pipeline. The sampler runs only when the pipeline is fitted, not when it predicts on validation or test data.

### When should you use random oversampling?

- The dataset is small.
- The minority class has useful examples but too few of them.
- You want a simple comparison against class weighting.

The downside is that repeating the same observations can increase overfitting.

## 7. Technique 3 — SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE creates new minority-class samples by interpolating between nearby minority-class observations.

Suppose two minority-class points are \\(x_i\\) and \\(x_j\\). SMOTE generates a synthetic point using:

\\[ x\_{\text{new}}=x_i+\lambda(x_j-x_i) \\]

where \\(\lambda\\) is a random value between 0 and 1.

In simple terms, it creates a point between two neighboring minority-class samples.

### Implement SMOTE properly

```
from imblearn.over_sampling import SMOTEfrom imblearn.pipeline import Pipeline as ImbPipelinefrom sklearn.linear_model import LogisticRegressionfrom sklearn.preprocessing import StandardScalersmote_model = ImbPipeline([    # Scale before distance-based neighbor calculations    ("scaler", StandardScaler()),    # Resample training data only    (        "sampler",        SMOTE(            sampling_strategy=0.20,            k_neighbors=5,            random_state=42,        ),    ),    ("classifier", LogisticRegression(        max_iter=2000,        random_state=42,    )),])smote_model.fit(X_train, y_train)evaluate_model(    smote_model,    X_val,    y_val,    title="SMOTE + logistic regression",)
```

Why is scaling before SMOTE important?

SMOTE uses distances between samples. If one feature ranges from 0 to 100,000 while another ranges from 0 to 1, the larger-scale feature can dominate the distance calculation.

Standardization reduces this issue.

### Important production considerations

SMOTE is not always appropriate.

- It can create unrealistic combinations of features.
- It may be unsuitable for categorical data in its standard form.
- It can blur class boundaries or create synthetic outliers.
- In financial transactions, synthetic records may not reflect valid transaction behavior.

For mixed numeric and categorical data, consider `SMOTENC`. For many tabular datasets, compare SMOTE with class weighting rather than assuming SMOTE will win.

## 8. Technique 4 — Random undersampling

Random undersampling removes some majority-class training observations.

```
from imblearn.under_sampling import RandomUnderSamplerfrom imblearn.pipeline import Pipeline as ImbPipelinefrom sklearn.ensemble import RandomForestClassifierundersampling_model = ImbPipeline([    (        "sampler",        RandomUnderSampler(            sampling_strategy=0.20,            random_state=42,        ),    ),    (        "classifier",        RandomForestClassifier(            n_estimators=300,            min_samples_leaf=2,            random_state=42,            n_jobs=-1,        ),    ),])undersampling_model.fit(X_train, y_train)evaluate_model(    undersampling_model,    X_val,    y_val,    title="Random undersampling + random forest",)
```

The ratio means the minority-to-majority ratio after sampling is 0.20.

### When is undersampling useful?

- The majority class is extremely large.
- Training time or memory is a concern.
- There is enough majority-class data to discard some without losing essential patterns.

The trade-off is that useful information from the majority class may be lost. In high-volume fraud systems, informed sampling can be preferable to blindly removing records.

## 9. Technique 5 — Tune the decision threshold

This is one of the most important techniques to understand.

A classifier often returns a probability such as:

```
probability_of_fraud = 0.32
```

With a threshold of `0.50`, the model predicts legitimate. With a threshold of `0.25`, it predicts fraud.

Lowering the threshold generally increases recall and can also increase false positives. Raising it often has the opposite effect.

You do not always need to resample a dataset to improve detection. You can tune the decision threshold using validation data.

### Select a threshold using validation F1

```
from sklearn.metrics import f1_score# Use probabilities from the validation split,# not probabilities from the test split.threshold_results = []for threshold in np.arange(0.05, 0.96, 0.01):    val_predictions = (        weighted_val_probabilities >= threshold    ).astype(int)    threshold_results.append({        "threshold": float(threshold),        "precision": precision_score(            y_val, val_predictions, zero_division=0        ),        "recall": recall_score(            y_val, val_predictions, zero_division=0        ),        "f1": f1_score(            y_val, val_predictions, zero_division=0        ),    })threshold_df = pd.DataFrame(threshold_results)best_row = threshold_df.loc[threshold_df["f1"].idxmax()]best_threshold = float(best_row["threshold"])print("Best validation threshold:", best_threshold)print(best_row)
```

Now evaluate the model using the chosen threshold on the untouched test data:

```
test_probabilities = weighted_model.predict_proba(X_test)[:, 1]test_predictions = (    test_probabilities >= best_threshold).astype(int)print("Test precision:",      precision_score(y_test, test_predictions, zero_division=0))print("Test recall:",      recall_score(y_test, test_predictions, zero_division=0))print("Test F1:",      f1_score(y_test, test_predictions, zero_division=0))print("Test confusion matrix:")print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))
```

The test set is used only after the threshold has been chosen. If you repeatedly adjust the threshold based on test results, the test set gradually becomes part of your model-selection process.

### Is maximizing F1 always the right choice?

No. Consider a fraud system:

- Missing a fraud transaction may cost ₹50,000.
- Investigating a false alert may cost ₹200.

In that situation, the best threshold may be the one that minimizes expected cost, subject to the available investigation capacity.

A simplified expected cost is:

\\[ \text{Cost}=C\_{FN}\cdot FN+C\_{FP}\cdot FP \\]

Where \\(C\_{FN}\\) is the cost of a missed positive and \\(C\_{FP}\\) is the cost of a false positive.

Use business costs, precision/recall constraints, or investigation capacity to select the threshold when these matter more than F1.

## 10. Compare the techniques fairly

After running the examples, you should compare them on the same validation data.

Here is a compact comparison of the approaches:

| Technique           | Changes training data?    | Main benefit                           | Main risk                                             |
| ------------------- | ------------------------- | -------------------------------------- | ----------------------------------------------------- |
| Class weighting     | No                        | Simple, preserves observations         | May increase false positives                          |
| Random oversampling | Duplicates records        | More minority-class influence          | Overfitting to duplicates                             |
| SMOTE               | Creates synthetic records | Adds minority-class variety            | Unrealistic synthetic samples                         |
| Undersampling       | Removes records           | Faster training                        | Loses majority-class information                      |
| Threshold tuning    | No                        | Aligns predictions with business goals | Trade-off between false positives and false negatives |

Do not decide that SMOTE is better merely because it increases recall. Compare precision, recall, average precision, confusion matrices, and business cost. Evaluate each candidate on the same untouched validation set.

For a more systematic experiment, you can build a table from the validation predictions:

```
from sklearn.metrics import average_precision_scoremodels = {    "Class weighting": weighted_model,    "Random oversampling": oversampling_model,    "SMOTE": smote_model,    "Undersampling": undersampling_model,}comparison = []for name, model in models.items():    probabilities = model.predict_proba(X_val)[:, 1]    predictions = (probabilities >= 0.5).astype(int)    comparison.append({        "model": name,        "precision": precision_score(            y_val, predictions, zero_division=0        ),        "recall": recall_score(            y_val, predictions, zero_division=0        ),        "f1": f1_score(            y_val, predictions, zero_division=0        ),        "average_precision": average_precision_score(            y_val, probabilities        ),    })comparison_df = pd.DataFrame(comparison)print(comparison_df.sort_values("average_precision", ascending=False))
```

This comparison uses a common threshold of 0.5 to make the initial comparison straightforward. For final selection, tune thresholds using validation data according to the chosen business objective. Also note that the models' probabilities may not be equally calibrated.

## 11. How I would handle imbalanced datasets in a production ML system

For a real financial or enterprise ML application, I would follow this workflow:

### 1

Understand the data

Measure class counts, prevalence, label quality, and whether the imbalance is expected.

### 2

Choose a baseline

Train a simple model and a majority-class baseline; measure minority-class performance.

### 3

Split correctly

Use stratified splits for independent observations, or time-based splits for future-facing financial prediction.

### 4

Compare interventions

Test class weights, sampling strategies, feature improvements, and suitable model families.

### 5

Tune on validation data

Select the operating threshold using expected cost, minimum recall, precision requirements, or investigation capacity.

### 6

Evaluate once on test data

Report precision, recall, F1, PR-AUC, confusion matrix, and confidence intervals where appropriate.

### 7

Deploy and monitor

Track class prevalence, precision/recall after labels arrive, data drift, false alerts, latency, and cost.

### Important production details

- Prevent leakage: Fit preprocessing, sampling, and feature selection only on training folds. Use an imbalanced-learn pipeline with cross-validation.
- Respect time: For transactions or financial forecasting, a chronological split may be more realistic than a random split.
- Handle probability calibration: Class weighting and resampling can change the relationship between predicted probabilities and real-world prevalence. If probabilities drive financial decisions, validate calibration and consider calibration methods on appropriate held-out data.
- Avoid data contamination: If the same customer, account, or transaction family appears repeatedly, consider group-based splits.
- Monitor delayed labels: Fraud labels may arrive days or weeks after a transaction. Track performance when verified labels become available, not just immediate model scores.
- Consider the base rate: Precision depends strongly on the actual positive-class prevalence. Results on an artificially balanced test set may not reflect production performance.
- Use human review when needed: In a high-risk financial system, route uncertain or high-impact predictions to an investigation or approval workflow.

## 12. Senior ML engineer interview answer

If an interviewer asks, “How do you handle an imbalanced dataset?”, a strong answer would be:

> I first inspect class prevalence, label quality, and the business cost of false positives versus false negatives. I establish a baseline and select metrics such as recall, precision, F1, average precision, and expected business cost instead of relying on accuracy alone.
>
> I create leakage-safe train, validation, and test splits. I start with class weighting, then compare random oversampling, SMOTE, and undersampling where appropriate. I tune the decision threshold on validation data to satisfy business constraints, and I evaluate the selected model on an untouched test set.
>
> For production, I validate probability calibration, monitor performance as labels arrive, and track changes in prevalence and model behavior. For temporal financial data, I use time-aware validation and ensure that no future information leaks into training.

The most important takeaway is that class imbalance is not just a data-preprocessing problem. It is a combination of data quality, learning strategy, evaluation, threshold selection, and business decision-making.