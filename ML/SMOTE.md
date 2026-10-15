# SMOTE in Machine Learning — Step-by-Step with Python Code

SMOTE is a technique used to handle imbalanced datasets, where one class has many more examples than another.

For example, imagine you are building a financial fraud detection model:

- 99,000 transactions are legitimate.
- 1,000 transactions are fraudulent.

Your model could predict “legitimate” for every transaction and achieve 99% accuracy while detecting zero fraud cases.

SMOTE helps by creating synthetic minority-class training examples so that the model has more examples to learn from.

## 1. What is SMOTE?

SMOTE stands for Synthetic Minority Over-sampling Technique.

It generates new synthetic samples for the minority class by interpolating between existing minority-class examples and their nearest neighbors.

[Sample generator used in SMOTE-like samplers — Version 0.15.dev0](https://images.openai.com/static-rsc-4/iT1lRc7Im6200x2JlBpG59JWB4Wf7hZUqeXPbq0_XERSTnLFGBLJHA1WtuXNPBZlA56w1q4xSSo-GbcH8Y37dXKepqzRpHEaHn2DWkOeDjIOAsJW2Jcb7tlDdSbAUsyk0IrPVhg4SWwLk45excsonpqql1xughEqz7wfcPbfXhA?purpose=inline)

[imbalanced-learn.org](https://imbalanced-learn.org/dev/auto_examples/over-sampling/plot_illustration_generation_sample.html)

Suppose two minority-class examples have these features:

- Transaction A: amount = 100, account age = 10
- Transaction B: amount = 200, account age = 20

SMOTE can generate a synthetic example between them, such as:

\- Synthetic transaction: amount = 140, account age = 14

This is a simplified numerical illustration. Actual SMOTE selects a minority example, finds one of its minority-class nearest neighbors, and interpolates between their feature vectors.

The formula is:

\\[ x\_{\text{new}}=x_i+\lambda(x\_{neighbor}-x_i) \\]

where:

- \\(x_i\\) is an existing minority-class sample.
- \\(x\_{neighbor}\\) is one of its minority-class nearest neighbors.
- \\(\lambda\\) is a random number between 0 and 1.
- \\(x\_{\text{new}}\\) is the generated synthetic sample.

SMOTE does not simply duplicate the minority samples; it creates new points between existing ones.

## 2. Understand the problem before using SMOTE

Let's create a synthetic classification dataset.

```
from sklearn.datasets import make_classificationfrom collections import CounterX, y = make_classification(    n_samples=10_000,    n_features=10,    n_informative=5,    n_redundant=2,    weights=[0.95, 0.05],    random_state=42)print("Dataset shape:", X.shape)print("Class distribution:", Counter(y))
```

Example output:

```
Dataset shape: (10000, 10)
Class distribution: Counter({0: 9459, 1: 541})
```

Here:

- `0` represents the majority class.
- `1` represents the minority class.
- Only about 5% of examples belong to class `1`.

This is an imbalanced classification problem.

In a real fraud detection dataset, class `1` might represent fraudulent transactions. In a medical dataset, it might represent a rare condition.

## 3. First, build a model without SMOTE

Before applying SMOTE, we need a baseline model. This helps us understand whether oversampling actually improves performance.

The most important rule is:

Split the data into training and testing sets before applying SMOTE. Never apply SMOTE to the complete dataset before splitting.

```
from sklearn.model_selection import train_test_splitfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import (    classification_report,    confusion_matrix,    average_precision_score)X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42)print("Training:", Counter(y_train))print("Testing:", Counter(y_test))model = LogisticRegression(max_iter=1000)model.fit(X_train, y_train)y_pred = model.predict(X_test)y_prob = model.predict_proba(X_test)[:, 1]print(confusion_matrix(y_test, y_pred))print(classification_report(y_test, y_pred, zero_division=0))print(    "Average precision:",    average_precision_score(y_test, y_prob))
```

### Why use `stratify=y`?

It preserves approximately the same class proportions in the training and test sets.

For example, if the full dataset has 5% minority examples, both sets should contain roughly 5% minority examples.

### Why not rely on accuracy?

Consider these predictions on 1,000 transactions:

| Metric                                       | Result |
| -------------------------------------------- | ------ |
| Legitimate transactions                      | 950    |
| Fraudulent transactions                      | 50     |
| Correctly classified legitimate transactions | 950    |
| Fraud detected                               | 0      |
| Accuracy                                     | 95%    |
| Fraud recall                                 | 0%     |

The model achieves 95% accuracy but fails to detect any fraud.

For imbalanced classification, we should examine:

- Precision: Of the transactions flagged as fraud, how many were actually fraud?
- Recall: Of all actual fraud cases, how many did the model detect?
- F1-score: A balance between precision and recall.
- Average precision / PR-AUC: Useful for evaluating minority-class ranking, especially when the positive class is rare.

## 4. Install SMOTE

SMOTE is provided by the `imbalanced-learn` Python library.

```
python -m pip install imbalanced-learn scikit-learn
```

Import it:

```
from imblearn.over_sampling import SMOTE
```

Now let's apply it to the training data.

```
smote = SMOTE(    sampling_strategy="auto",    k_neighbors=5,    random_state=42)X_train_resampled, y_train_resampled = smote.fit_resample(    X_train,    y_train)print("Before SMOTE:", Counter(y_train))print("After SMOTE:", Counter(y_train_resampled))
```

For binary classification, `sampling_strategy="auto"` oversamples the minority class until its count matches the majority class.

Your output will look similar to this:

```
Before SMOTE: Counter({0: 7567, 1: 433})
After SMOTE:  Counter({0: 7567, 1: 7567})
```

The exact counts depend on the dataset split.

What happened?

- The majority-class examples remain unchanged.
- Synthetic minority-class examples are added to the training set.
- The test set is untouched.
- SMOTE has not learned anything from the test data.

Now train the model on the resampled data:

```
model = LogisticRegression(max_iter=1000)model.fit(X_train_resampled, y_train_resampled)y_pred = model.predict(X_test)y_prob = model.predict_proba(X_test)[:, 1]print(confusion_matrix(y_test, y_pred))print(classification_report(y_test, y_pred, zero_division=0))print(    "Average precision:",    average_precision_score(y_test, y_prob))
```

The results may show higher recall for the minority class, but precision may decrease because the model flags more transactions as potentially fraudulent. This is not guaranteed: you must compare the results against the baseline on the same untouched test set.

Also, SMOTE does not necessarily improve average precision or overall business performance. It is one experiment to evaluate, not a universal solution.

## 5. The production-safe approach: SMOTE inside an ML pipeline

The previous example works for a simple experiment. However, a senior ML engineer should pay attention to a subtle problem: SMOTE uses nearest-neighbor distances.

If features have very different scales, distance calculations can be misleading.

For example:

- Annual income: 1,200,000
- Account age: 5
- Number of transactions: 20

Income may dominate the distance calculation simply because its numerical values are much larger.

We can scale the features before SMOTE. However, scaling must also avoid data leakage.

### Correct pipeline architecture

Training data only

Original, imbalanced examples

StandardScaler

Learn mean and standard deviation from the training fold

SMOTE

Generate minority-class samples for training

Classifier

Train the model on the resampled data

Untouched test data

Only transformed using the scaler fitted on training data; never resampled.

Here is the recommended implementation using `imblearn.pipeline.Pipeline`:

```
from imblearn.pipeline import Pipelinefrom imblearn.over_sampling import SMOTEfrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import classification_report, average_precision_scoresmote_pipeline = Pipeline(steps=[    ("scaler", StandardScaler()),    ("smote", SMOTE(        sampling_strategy="auto",        k_neighbors=5,        random_state=42    )),    ("classifier", LogisticRegression(max_iter=1000))])# Only training data is used to fit the pipeline.smote_pipeline.fit(X_train, y_train)# The pipeline does not run SMOTE on the test data.y_pred = smote_pipeline.predict(X_test)y_prob = smote_pipeline.predict_proba(X_test)[:, 1]print(classification_report(y_test, y_pred, zero_division=0))print(    "Average precision:",    average_precision_score(y_test, y_prob))
```

Notice that we use `imblearn.pipeline.Pipeline`, not `sklearn.pipeline.Pipeline`. The imbalanced-learn pipeline supports resampling steps such as SMOTE during fitting.

When `predict()` or `predict_proba()` runs on the test set, the pipeline scales the test features and sends them to the trained classifier. It does not create synthetic test examples.

### Why is this production-grade practice?

It makes preprocessing and resampling part of one reproducible model artifact. The same steps can be evaluated consistently during cross-validation and later deployed with the trained model.

One important distinction: the scaler is needed during inference, but SMOTE is a training-time operation and is not needed for individual production predictions.

## 6. The critical rule for cross-validation

A common mistake is to apply SMOTE once to the entire training dataset and then run cross-validation on the resampled data. Synthetic samples can be derived from examples that end up in different folds, contaminating the evaluation.

Instead, put SMOTE inside the pipeline and pass the pipeline to cross-validation. Each fold will fit its own scaler and generate synthetic samples using only that fold's training portion.

```
from sklearn.model_selection import StratifiedKFold, cross_validatefrom imblearn.pipeline import Pipelinefrom imblearn.over_sampling import SMOTEfrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionpipeline = Pipeline(steps=[    ("scaler", StandardScaler()),    ("smote", SMOTE(random_state=42, k_neighbors=5)),    ("classifier", LogisticRegression(max_iter=1000))])cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42)scores = cross_validate(    pipeline,    X_train,    y_train,    cv=cv,    scoring={        "precision": "precision",        "recall": "recall",        "f1": "f1",        "average_precision": "average_precision"    },    n_jobs=-1)for metric in (    "precision",    "recall",
```

This provides a more robust estimate of model performance than evaluating one training split alone. After selecting the approach using cross-validation, fit the final pipeline on the training data and evaluate it on the untouched test set once.

## 7. Important SMOTE parameters

| Parameter           | Meaning                                            | Practical guidance                                                                                                  |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `sampling_strategy` | Desired minority-class proportion after resampling | Start with `"auto"` as a baseline; also test partial oversampling                                                   |
| `k_neighbors`       | Number of minority-class neighbors considered      | Default is 5; small minority datasets may require fewer                                                             |
| `random_state`      | Makes random sampling reproducible                 | Set a fixed seed for experiments                                                                                    |
| `n_jobs`            | Parallelism parameter                              | Do not rely on it for current SMOTE versions; it is not a supported SMOTE constructor parameter in current releases |

For binary classification, `sampling_strategy=0.5` means the minority-class count after resampling will be half the majority-class count.

For example:

```
smote = SMOTE(    sampling_strategy=0.5,    k_neighbors=3,    random_state=42)
```

You don't always need a 50:50 dataset. Sometimes oversampling to a 1:2 ratio works better because excessive oversampling can increase false positives.

Also, `k_neighbors` must be smaller than the number of minority-class samples available in each training fold.

## 8. What if the dataset contains categorical features?

Standard SMOTE is designed for numerical feature vectors. Applying it directly to encoded categorical data can produce invalid fractional category values.

For example, suppose `employment_type` is encoded as:

- Salaried = 0
- Self-employed = 1
- Unemployed = 2

Standard SMOTE could generate `1.4`, which is not a valid employment category.

For mixed numerical and categorical data, consider SMOTENC.

```
from imblearn.over_sampling import SMOTENC# Example column positions:# 0 = annual_income (numeric)# 1 = account_age (numeric)# 2 = employment_type (categorical)# 3 = city (categorical)categorical_indices = [2, 3]smotenc = SMOTENC(    categorical_features=categorical_indices,    sampling_strategy="auto",    random_state=42)X_resampled, y_resampled = smotenc.fit_resample(X_train, y_train)
```

This example assumes `X_train` contains the original numerical and categorical columns, represented in a form SMOTENC can handle. Do not apply ordinary scaling or one-hot encoding indiscriminately before this operation; design the preprocessing order carefully so the categorical columns remain identifiable to SMOTENC. For a mixed-type production pipeline, `ColumnTransformer` and a deliberately constructed imbalanced-learn pipeline are often appropriate.

## 9. When should you avoid SMOTE?

SMOTE is useful, but it is not always the right solution.

- Text classification: Interpolating TF-IDF vectors can produce synthetic feature vectors that do not correspond to realistic text. Consider class weights or suitable text augmentation instead.
- Categorical-heavy datasets: Prefer SMOTENC or compare against class weighting.
- Extremely rare fraud: Synthetic examples may not represent the true diversity of fraud patterns.
- Noisy minority labels: SMOTE may amplify mislabeled or anomalous minority samples.
- Overlapping classes: Synthetic points can be created in regions where the classes overlap, increasing false positives.

Always compare SMOTE against a baseline with `class_weight="balanced"` and, when appropriate, a model-specific class-weighting or cost-sensitive approach.

## 10. A senior ML engineer's evaluation checklist

Before shipping a model that uses SMOTE, verify the following:

- Split data before oversampling.
- Apply SMOTE only to training folds.
- Scale numerical features before distance-based oversampling where appropriate.
- Use SMOTENC or another suitable approach for mixed data.
- Compare against a no-SMOTE baseline.
- Evaluate precision, recall, F1, average precision and the business cost of errors.
- Tune the prediction threshold using validation data when needed.
- Evaluate on a realistic, naturally imbalanced test set.
- Monitor production precision, recall and class-distribution drift.

For a financial fraud system, missing a fraud case and incorrectly blocking a legitimate transaction have different costs. Your final model and threshold should reflect those costs, not merely produce a balanced training dataset.

## Interview answer

What is SMOTE, and how do you use it in production?

SMOTE is a synthetic oversampling technique that generates minority-class samples by interpolating between a minority example and its nearest minority-class neighbors. I use it to address class imbalance, but I first establish a baseline and compare against class-weighted models. I keep SMOTE inside an imbalanced-learn pipeline so that resampling happens only during training, including independently within each cross-validation fold. I evaluate using precision, recall, F1 and average precision on naturally distributed validation and test data. I also check whether the generated samples are meaningful for the feature types and the business domain.

The key principle: SMOTE changes the training distribution; it does not make the real-world class distribution balanced. Your validation and test sets should still represent the problem you expect the model to solve in production.