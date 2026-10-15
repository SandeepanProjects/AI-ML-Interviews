# Data Leakage in Machine Learning — Complete Explanation With Python Code

Data leakage is one of the most important problems to understand when building reliable machine learning systems. A model can achieve 99% accuracy during testing because of leakage, then perform poorly when deployed in production.

For a senior ML or AI engineer, the key skill is not just knowing what leakage means, but identifying exactly where it occurs in data preparation, feature engineering, model training, validation, and deployment.

## 1. What is data leakage?

Data leakage occurs when a model gets access to information during training or evaluation that would not legitimately be available when making the actual prediction.

This gives the model an unfair advantage and can produce overly optimistic evaluation results.

Imagine you're building a model to predict whether a customer will default on a loan.

Your dataset contains:

| Feature                  | Example    | Available at prediction time?         |
| ------------------------ | ---------- | ------------------------------------- |
| Annual income            | ₹12,00,000 | Yes                                   |
| Credit score             | 720        | Yes                                   |
| Existing debt            | ₹4,00,000  | Yes                                   |
| Previous missed payments | 2          | Yes                                   |
| Loan recovery status     | Recovered  | No, if recorded after default         |
| Collection date          | 2026-08-10 | No, if it occurs after the prediction |

If you train a model using `loan_recovery_status`, it may learn to predict default from information that becomes known only after the default occurs.

The model might score exceptionally well in testing, but it cannot use that information to make an earlier prediction.

This is target leakage.

## 2. How leakage makes a model look better than it really is

Training dataset

Contains information unavailable at prediction time

Model learns leaked information

An invalid shortcut makes prediction appear easy

Misleading evaluation

## 99% accuracy

The evaluation does not represent real deployment conditions

The model hasn't necessarily learned the underlying problem. It may have learned an artifact of how the dataset was constructed.

There are several common forms of leakage, and each needs a different prevention strategy.

## 3. The main types of data leakage

| Type                      | Example                                                           | Prevention                                                           |
| ------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| Target leakage            | Using a post-default recovery status to predict default           | Exclude features unavailable at prediction time                      |
| Preprocessing leakage     | Scaling or imputing the entire dataset before splitting           | Fit transformations on training data only                            |
| Feature-selection leakage | Selecting features using the full dataset before cross-validation | Put feature selection inside the pipeline                            |
| Train-test contamination  | Duplicate or related records appear in both sets                  | Use group-aware splitting where appropriate                          |
| Temporal leakage          | Training on future information to predict past outcomes           | Use chronological splits and point-in-time features                  |
| Evaluation leakage        | Repeatedly tuning the model using test-set results                | Use validation data for model selection and reserve a final test set |

Let's examine these with actual code.

## 4. Example 1: Target leakage

Suppose we are predicting whether a loan will default.

We have three features:

- `income`: known before the loan outcome.
- `credit_score`: known before the loan outcome.
- `recovery_status`: recorded after default.

The last feature reveals information about the outcome.

### Step 1: Create a dataset with a deliberately leaked feature

```
import numpy as npimport pandas as pdfrom sklearn.model_selection import train_test_splitfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import accuracy_scorerng = np.random.default_rng(42)n_samples = 5000income = rng.normal(80000, 25000, n_samples)credit_score = rng.normal(680, 60, n_samples)# Simulate the actual default outcomedefault_probability = 1 / (    1 + np.exp(        -(            -2.0            - 0.00001 * (income - 80000)            - 0.008 * (credit_score - 680)        )    ))defaulted = rng.binomial(1, default_probability)# Deliberately leaked: recorded after the outcomerecovery_status = defaulted.copy()df = pd.DataFrame({    "income": income,    "credit_score": credit_score,    "recovery_status": recovery_status,    "defaulted": defaulted,})
```

This is synthetic educational data. In this example, `recovery_status` is intentionally constructed to reveal the target exactly. It would be invalid as a predictor for a model that must predict default before it occurs.

### Step 2: Train a model using the leaked feature

```
X = df[[    "income",    "credit_score",    "recovery_status",]]y = df["defaulted"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)model = RandomForestClassifier(    n_estimators=100,    random_state=42,)model.fit(X_train, y_train)predictions = model.predict(X_test)print(    "Accuracy with leakage:",    accuracy_score(y_test, predictions),)
```

The accuracy should be 1.0 in this synthetic example because the leaked feature exactly equals the target.

This score is meaningless as a measure of the model's ability to predict future defaults.

### Step 3: Remove the leaked feature

```
X = df[[    "income",    "credit_score",]]y = df["defaulted"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)model = RandomForestClassifier(    n_estimators=100,    random_state=42,)model.fit(X_train, y_train)predictions = model.predict(X_test)print(    "Accuracy without leakage:",    accuracy_score(y_test, predictions),)
```

Now the model must learn from the legitimate predictors rather than a direct copy of the outcome. Its score should be lower, reflecting the difficulty of the task.

Senior-engineer lesson: Never decide whether a feature is valid based only on how predictive it is. First determine whether that feature would actually be available at the exact moment the prediction is made.

## 5. Example 2: Preprocessing leakage — scaling before splitting

This is one of the most common mistakes beginners make.

Suppose you have 10,000 records and want to standardize each feature.

The incorrect approach is:

1. Calculate means and standard deviations using all 10,000 records.
2. Split into training and test sets.
3. Train and evaluate the model.

The problem is that the test-set distribution influenced the preprocessing parameters.

Even though the scaler doesn't use the target labels, information from the test set has influenced the model-development process.

### Incorrect code

```
from sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import accuracy_scoreX, y = make_classification(    n_samples=5000,    n_features=15,    n_informative=8,    random_state=42,)# WRONG: learn scaling parameters from all observationsscaler = StandardScaler()X_scaled = scaler.fit_transform(X)# Split after preprocessingX_train, X_test, y_train, y_test = train_test_split(    X_scaled,    y,    test_size=0.20,    stratify=y,    random_state=42,)model = LogisticRegression(max_iter=2000)model.fit(X_train, y_train)predictions = model.predict(X_test)print("Accuracy:", accuracy_score(y_test, predictions))
```

The reported accuracy may look reasonable, but the workflow is not a clean simulation of training on past data and predicting unseen observations.

For a simple random split from the same distribution, the impact of scaling leakage might be small. The important issue is the invalid methodology, which can become more consequential under distribution shift or more complex preprocessing.

### Correct code

Split first. Fit the scaler only on the training data, then transform the test data.

```
X, y = make_classification(    n_samples=5000,    n_features=15,    n_informative=8,    random_state=42,)X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)scaler = StandardScaler()# Learn mean and standard deviation from training dataX_train_scaled = scaler.fit_transform(X_train)# Reuse training parameters on test dataX_test_scaled = scaler.transform(X_test)model = LogisticRegression(max_iter=2000)model.fit(X_train_scaled, y_train)predictions = model.predict(X_test_scaled)print("Accuracy:", accuracy_score(y_test, predictions))
```

Notice the difference:

- `fit_transform()` learns and applies the transformation.
- `transform()` applies a previously learned transformation without fitting it again.

That distinction applies to scaling, imputation, feature selection, PCA, and many other preprocessing steps.

## 6. The safest solution: use a scikit-learn pipeline

In production ML, you should usually combine preprocessing and the model into a single pipeline.

```
from sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import classification_reportpipeline = Pipeline([    ("scaler", StandardScaler()),    ("classifier", LogisticRegression(max_iter=2000)),])pipeline.fit(X_train, y_train)predictions = pipeline.predict(X_test)print(classification_report(y_test, predictions))
```

Why is this safer?

When you call `pipeline.fit(X_train, y_train)`, the pipeline:

1. Learns the scaler parameters from `X_train`.
2. Transforms `X_train`.
3. Fits logistic regression on the transformed data.

When you call `pipeline.predict(X_test)`, it uses the already-fitted scaler and classifier.

The pipeline also works correctly with cross-validation because each fold gets its own fitted preprocessing steps.

Production rule: Package all learned preprocessing steps together with the trained model so training and inference use the same transformation logic.

## 7. Example 3: Data leakage during missing-value imputation

Imagine a feature such as annual income has missing values.

A common approach is to replace missing values with the median.

The incorrect approach calculates the median before splitting the dataset.

### Incorrect code

```
import pandas as pdfrom sklearn.impute import SimpleImputerfrom sklearn.model_selection import train_test_splitX = pd.DataFrame({    "income": [40000, 50000, None, 60000, 70000,               80000, None, 90000, 100000, 110000],    "credit_score": [620, 650, 670, None, 710,                     730, 750, 760, None, 800],})y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]# WRONG: calculate medians using all rowsimputer = SimpleImputer(strategy="median")X_imputed = imputer.fit_transform(X)X_train, X_test, y_train, y_test = train_test_split(    X_imputed,    y,    test_size=0.20,    random_state=42,)
```

The imputation values have been learned from both training and test records.

### Correct code: imputation inside a pipeline

```
from sklearn.impute import SimpleImputerfrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinepipeline = Pipeline([    ("imputer", SimpleImputer(strategy="median")),    ("scaler", StandardScaler()),    ("classifier", LogisticRegression(max_iter=2000)),])X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)pipeline.fit(X_train, y_train)predictions = pipeline.predict(X_test)print(predictions)
```

Here the imputer learns the median from training data only. The same median is used for validation, testing, and inference.

One important nuance: if the median is calculated using the full dataset without the labels, the resulting leakage may be unsupervised and its practical effect may be small in some settings. Nevertheless, it violates a clean evaluation design and can cause misleading results when the data distribution changes.

## 8. Example 4: Feature-selection leakage

Suppose you have 1,000 features and want to keep the 20 most predictive ones.

A common mistake is to select those 20 features using the entire dataset, then perform cross-validation on the reduced dataset.

The validation folds have already influenced which features were selected. The cross-validation results are therefore biased.

### Incorrect pattern

```
from sklearn.feature_selection import SelectKBest, f_classiffrom sklearn.model_selection import cross_val_scorefrom sklearn.linear_model import LogisticRegression# WRONG: feature selection occurs before cross-validationselector = SelectKBest(score_func=f_classif, k=20)X_selected = selector.fit_transform(X, y)scores = cross_val_score(    LogisticRegression(max_iter=2000),    X_selected,    y,    cv=5,    scoring="accuracy",)print("CV accuracy:", scores.mean())
```

### Correct pattern: selection inside the pipeline

```
from sklearn.feature_selection import SelectKBest, f_classiffrom sklearn.pipeline import Pipelinefrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import StratifiedKFold, cross_val_scorepipeline = Pipeline([    ("selector", SelectKBest(score_func=f_classif, k=20)),    ("classifier", LogisticRegression(max_iter=2000)),])cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42,)scores = cross_val_score(    pipeline,    X,    y,    cv=cv,    scoring="accuracy",)print("Leakage-safe CV accuracy:", scores.mean())
```

The selector is now fitted independently within each training fold.

The same principle applies to PCA, SMOTE, normalization, feature engineering that learns parameters from data, and other data-dependent transformations.

For resampling, use `imblearn.pipeline.Pipeline` so sampling happens only during fitting of each training fold.

## 9. Example 5: Temporal leakage in financial machine learning

Temporal leakage is especially important in financial applications, fraud detection, customer churn, and market forecasting.

Suppose you want to predict whether a customer will default in the next 30 days.

You have monthly records from January through December.

A random train-test split can allow the model to train on later months and evaluate on earlier months. That may be inappropriate if the real deployment task is to predict the future using the past.

For example:

- Training data: January through September.
- Test data: October through December.

This better simulates a future-facing prediction problem.

### Incorrect approach: random splitting of time-dependent records

```
from sklearn.model_selection import train_test_split# Assume df contains a "date" column and a "defaulted" targetX = df.drop(columns=["defaulted"])y = df["defaulted"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    random_state=42,)
```

Random splitting is not automatically wrong for every time-stamped dataset. It is problematic when it breaks the temporal structure needed to simulate real deployment, or when future information is included in features.

### Correct approach: chronological split

```
import pandas as pddf["date"] = pd.to_datetime(df["date"])df = df.sort_values("date").reset_index(drop=True)split_index = int(len(df) * 0.80)train_df = df.iloc[:split_index]test_df = df.iloc[split_index:]X_train = train_df.drop(columns=["defaulted", "date"])y_train = train_df["defaulted"]X_test = test_df.drop(columns=["defaulted", "date"])y_test = test_df["defaulted"]print(    "Train period:",    train_df["date"].min(),    "to",    train_df["date"].max(),)print(    "Test period:",    test_df["date"].min(),    "to",    test_df["date"].max(),)
```

This assumes that rows are ordered by the relevant prediction timeline and that the labels are correctly aligned.

In a real financial system, you must also ensure that each feature is computed only from information available at the prediction timestamp. A chronological split alone does not fix a feature that already contains future information.

For example, if the model predicts default as of 1 October, the following would be invalid:

- Recovery status recorded on 15 October.
- Total payments calculated through the end of October.
- A customer risk score updated after the default event.

A further improvement is a label horizon and embargo: if outcomes span 30 days, you may need to exclude or carefully handle observations whose outcome windows overlap the training/test boundary.

## 10. Example 6: Group leakage and duplicate customers

Imagine a healthcare model predicting disease risk or a financial model predicting customer default. The same customer may have many rows.

If records for the same customer appear in both training and test sets, the model might effectively recognize the customer rather than generalize to unseen customers.

This is called group leakage or train-test contamination.

For example:

| Customer ID | Month    | Defaulted |
| ----------- | -------- | --------- |
| C100        | January  | 0         |
| C100        | February | 0         |
| C100        | March    | 1         |
| C200        | January  | 0         |

If January and February for customer C100 are in training while March is in testing, the evaluation may not represent performance on entirely new customers.

Whether this is leakage depends on the real deployment goal. If you predict future outcomes for existing customers, a temporal split may be appropriate. If you need to generalize to new customers, keep customers separate across the splits.

### Correct approach: group-aware splitting

```
from sklearn.model_selection import GroupShuffleSplit# Assume:# X contains model input features# y contains the target# customer_ids contains one customer ID per rowsplitter = GroupShuffleSplit(    n_splits=1,    test_size=0.20,    random_state=42,)train_idx, test_idx = next(    splitter.split(X, y, groups=customer_ids))X_train = X.iloc[train_idx]X_test = X.iloc[test_idx]y_train = y.iloc[train_idx]y_test = y.iloc[test_idx]train_customers = set(customer_ids.iloc[train_idx])test_customers = set(customer_ids.iloc[test_idx])assert train_customers.isdisjoint(test_customers)print("Training rows:", len(X_train))print("Testing rows:", len(X_test))
```

This keeps customers separate across the splits. When both group separation and temporal ordering are necessary, design a split that enforces both requirements rather than assuming a random group split is sufficient.

## 11. Example 7: Target encoding leakage

Target encoding replaces a categorical value with a statistic derived from the target.

For example, suppose you want to predict loan default using `employer_name`. You might calculate the historical default rate for each employer.

If you calculate those rates using all rows, including validation or test labels, you leak target information into the features.

Even calculating the target mean using only training data can lead to overfitting for rare categories if each row's own label contributes heavily to its encoded value.

### Example of the wrong approach

```
# Assume df has "employer_name" and "defaulted"# WRONG: test labels influence the encodingdf["employer_default_rate"] = (    df.groupby("employer_name")["defaulted"]      .transform("mean"))
```

This computes each employer's default rate using all rows, including the outcomes you later intend to predict.

### Better approach

Use a target encoder that is fitted only on training data, with appropriate smoothing and cross-fitting. For example, scikit-learn provides `TargetEncoder` in newer versions:

```
from sklearn.preprocessing import TargetEncoderfrom sklearn.pipeline import Pipelinefrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import train_test_splitX = df[["employer_name"]]y = df["defaulted"]X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)pipeline = Pipeline([    (        "encoder",        TargetEncoder(            target_type="binary",            smooth="auto",            cv=5,            random_state=42,        ),    ),    ("classifier", LogisticRegression(max_iter=2000)),])pipeline.fit(X_train, y_train)predictions = pipeline.predict(X_test)print(predictions[:10])
```

`TargetEncoder` uses cross-fitting during `fit_transform()` to reduce leakage from a row's own target into its encoded training value. The pipeline learns the encoding using training data and applies the fitted mapping to the test data.

If you're using an older scikit-learn version, check whether `TargetEncoder` is available or upgrade to a compatible version.

## 12. Example 8: Repeatedly using the test set to tune the model

Leakage isn't always caused by a feature column. It can also occur through the model-development process.

Suppose you do the following:

1. Train a model.
2. Evaluate on the test set.
3. Change hyperparameters based on test performance.
4. Evaluate again.
5. Repeat 100 times.

You have gradually used the test set to guide model selection. The final test score may be overly optimistic because your choices have adapted to the test data.

### Correct approach

Use three distinct roles:

| Dataset                       | Role                                          |
| ----------------------------- | --------------------------------------------- |
| Training                      | Fit model parameters and preprocessing        |
| Validation / cross-validation | Compare models and tune hyperparameters       |
| Test                          | Final evaluation after decisions are complete |

For example, tune model hyperparameters using `GridSearchCV` and a pipeline, then evaluate the selected estimator on the test set once.

```
from sklearn.model_selection import GridSearchCV, StratifiedKFoldfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import classification_reportpipeline = Pipeline([    ("scaler", StandardScaler()),    ("classifier", LogisticRegression(max_iter=2000)),])param_grid = {    "classifier__C": [0.01, 0.1, 1.0, 10.0],    "classifier__class_weight": [None, "balanced"],}cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42,)search = GridSearchCV(    estimator=pipeline,    param_grid=param_grid,    scoring="f1",    cv=cv,    n_jobs=-1,    refit=True,)search.fit(X_train, y_train)print("Best parameters:", search.best_params_)print("Best CV F1:", search.best_score_)
```

This example assumes an independent, stratifiable dataset. For time-dependent or grouped data, use an appropriate time-series or group-aware cross-validation strategy instead.

## 13. How to detect data leakage in a real ML project

Use this checklist during code review.

## Leakage prevention checklist

0/10

Every feature is available at the exact time the prediction is made.

The target label is excluded from the input features.

The train/test split happens before fitting preprocessing steps.

Imputation, scaling, PCA, and feature selection are fitted only on training folds.

Oversampling and SMOTE happen only inside training folds.

Repeated customers, patients, or related records are split appropriately.

Temporal data is evaluated using a realistic chronological split.

Target encoding and aggregate features use only permitted historical information.

The test set is not repeatedly used to select models or thresholds.

The production feature-generation process matches the training process.

## 14. Senior ML engineer interview answer

If an interviewer asks, “What is data leakage, and how do you prevent it?”, you can answer:

> Data leakage occurs when training or evaluation uses information that would not legitimately be available at prediction time, resulting in overly optimistic performance estimates.
>
> I first check for target leakage by reviewing feature definitions, timestamps, and the business process that generates each field. Then I prevent preprocessing leakage by fitting imputers, scalers, feature selectors, and dimensionality-reduction steps inside a scikit-learn pipeline.
>
> I use cross-validation to tune models without contaminating the final test set. For repeated entities, I consider group-aware splits; for financial or time-dependent predictions, I use chronological validation and point-in-time feature generation. I also validate the training-serving feature pipeline to ensure that production predictions use only information available at the time of inference.

Remember these three rules:

1. Respect time: Never use information that becomes available after the prediction point.
2. Split before fitting: Learn preprocessing parameters using training data only.
3. Protect evaluation: Use validation data for model selection and reserve the test set for final evaluation.

These rules prevent many of the most common forms of leakage and help ensure that your ML model's reported performance is closer to what you can achieve in production.