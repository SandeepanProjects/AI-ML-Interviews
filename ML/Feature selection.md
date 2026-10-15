# Feature Selection in Machine Learning — From Fundamentals to Production-Level Code

Feature selection is one of the most important steps in building reliable machine learning models. It is especially useful when working with financial data, fraud detection, customer analytics, recommendation systems, and enterprise AI applications.

I'll explain it step by step, with runnable Python code, mathematical intuition, practical examples, and a production-style scikit-learn pipeline.

## 1. What is feature selection?

Feature selection is the process of choosing the most useful input variables for training a machine learning model and removing variables that add little useful information.

Suppose you are building a model to predict whether a customer will default on a loan.

Your dataset contains these features:

| Feature                   | Example    | Potential usefulness                       |
| ------------------------- | ---------- | ------------------------------------------ |
| Annual income             | ₹12,00,000 | Useful                                     |
| Credit score              | 720        | Highly useful                              |
| Existing debt             | ₹4,00,000  | Highly useful                              |
| Number of missed payments | 2          | Highly useful                              |
| Customer ID               | 1049281    | Usually not predictive in a meaningful way |
| Random reference number   | 783401     | Usually not useful                         |
| Duplicate income field    | ₹12,00,000 | May add redundant information              |

You might have 200 columns, but perhaps only 25 contribute meaningfully to predicting loan default.

Feature selection tries to identify those useful columns.

Original dataset

200 input features

Feature selection

Statistical tests, model importance, or regularization

Selected dataset

25 useful features, if validated by the selection method

The number 25 is illustrative, not a target you should automatically aim for. The best feature set is the one that performs well on unseen data and meets your operational requirements.

## 2. Feature selection vs. feature extraction vs. dimensionality reduction

These concepts are related but different.

| Technique                | What it does                                            | Example                                          |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------ |
| Feature selection        | Keeps a subset of existing columns                      | Keep `income`, `credit_score`, `missed_payments` |
| Feature extraction       | Creates new representations from existing columns       | Combine several measurements into a new feature  |
| Dimensionality reduction | Maps data to fewer dimensions, often using new features | PCA converts 100 columns into 10 components      |

For example, if you have 100 numeric features, PCA might transform them into 10 components. Those components are combinations of the original features; they are not necessarily individual business variables.

If interpretability is important, such as in loan risk or financial compliance, feature selection can be easier to explain than PCA.

## 3. Why do we need feature selection?

Feature selection can help with several problems:

- Overfitting: removes irrelevant features that can encourage a model to learn noise.
- Generalization: may improve performance on unseen data.
- Training cost: can reduce memory use and computation, particularly with wide datasets.
- Interpretability: helps engineers and domain experts understand which inputs matter.
- Data quality: can expose IDs, duplicate columns, leakage, or unstable features.

However, feature selection does not automatically improve every model. Tree-based models can already handle some irrelevant features reasonably well, and removing useful variables can reduce performance.

Also, feature selection is not the same as cleaning data. You still need appropriate handling of missing values, categorical variables, outliers, and leakage.

## 4. The three main approaches to feature selection

[Feature Selection Techniques in Machine Learning | by Saiprasadakkenpally | Jun, 2026 | Medium](https://images.openai.com/static-rsc-4/IGu0rEulw6gZy6rowHTi1zPcnbqZjj9gfT2zWvEf5DdnJGrSaZeFWHMcTpwCK4EhdhbfgCqzf-KLXuNRrYA2c_D3EhVTVPwV-GV0edKkc8g21tFYfVQtJGV6mqK29-shvrCu8aARaYaBcnQP3IQfwhA5p01n9kBa_ZltIPsLfKA?purpose=inline)

1\. Filter methods

Rank or remove features using statistics before fitting the final model. Examples: correlation, chi-square, ANOVA, mutual information, variance threshold.

[Learning scikit-learn -- An Introduction to Machine Learning in Python @ PyData Chicago 2016 - Speaker Deck](https://images.openai.com/static-rsc-4/0yjU_zOs0JiKuBYu5Pp09ypZAmVgY36otmr67gOP_-P7EO7m8MnrqFyfFSEzO-7-TgRjnDenEbRFVAPSJenbpwWkO--ShFltw36Rz0I2vK1s2uuwMwmWOhNS5Y4jWKIMZuy77NPFO2sZGLGI-0x6ZHZAYv-EIBMShrVKavWGfhY?purpose=inline)

2\. Wrapper methods

Repeatedly train and evaluate models with different feature subsets. Examples: Recursive Feature Elimination (RFE) and Sequential Feature Selection.

[How to Interpret LASSO Coefficients | MetricGate](https://images.openai.com/static-rsc-4/JmsCOoUWeFzC30VTRrrIaIBkcNlrEYngFHhHbumTSto5SPAeickoZfjOPuwMCB-KQTP41iaAAZ6eMIQpBwg8mixk2gMv7oF0kHwjPmkcxbMdWjz4MKmJIgp3nkMm3CCYnK-2OsJIKFB7sMfMLbSi7CVrRwDWQRvTZQlfZjoquVU?purpose=inline)

3\. Embedded methods

Select or prioritize features as part of model training. Examples: L1-regularized logistic regression and tree-based feature importance.

Let's implement each approach.

## 5. Create a dataset for our experiments

We'll create a synthetic customer-default dataset with useful features, redundant features, and irrelevant features. This lets us see how different selection methods work.

Install the required libraries:

```
pip install numpy pandas scikit-learn
```

Now create the dataset:

```
import numpy as npimport pandas as pdfrom sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitX_array, y_array = make_classification(    n_samples=5000,    n_features=20,    n_informative=6,    n_redundant=5,    n_repeated=0,    n_classes=2,    class_sep=1.0,    random_state=42,)feature_names = [f"feature_{i}" for i in range(20)]X = pd.DataFrame(X_array, columns=feature_names)y = pd.Series(y_array, name="defaulted")print("Dataset shape:", X.shape)print("Target distribution:")print(y.value_counts())
```

Here:

- `n_features=20`: 20 input columns.
- `n_informative=6`: six features are generated as informative to the target.
- `n_redundant=5`: five features are generated as linear combinations of informative features.
- The remaining features are not intentionally informative in the data-generating process.

Important: the generated dataset is synthetic. In a real dataset, you do not know which features are truly informative, and the usefulness of a feature can depend on which model and other features you use.

Split the data before selecting features:

```
X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)print("Training data:", X_train.shape)print("Test data:", X_test.shape)
```

We will use the training set to learn which features to select and reserve the test set for final evaluation.

Never select features using the complete dataset before splitting. That allows information from the test set to influence your selection.

## 6. Method 1 — Filter methods

Filter methods use statistical properties of the features and target to identify potentially useful columns. They are generally fast and can be useful when the dataset has hundreds or thousands of features.

### 6.1 Correlation-based feature selection

Correlation measures how strongly two numeric variables are associated. For a numeric target, Pearson correlation can help identify linear relationships. For a binary target encoded as 0 and 1, correlation can also measure linear association with the target.

```
# Use training data onlytrain_df = X_train.copy()train_df["target"] = y_traincorrelations = (    train_df.corr(numeric_only=True)["target"]    .drop("target")    .sort_values(key=lambda s: s.abs(), ascending=False))print(correlations)
```

You can inspect the strongest associations:

```
top_features = correlations.abs().head(8).index.tolist()print("Top 8 features by absolute correlation:")print(top_features)
```

What does this tell us?

A feature with a high absolute correlation may be useful for predicting the target. A feature with near-zero correlation may have little linear association.

But correlation alone is not sufficient:

- A feature can have a nonlinear relationship with the target and low Pearson correlation.
- Two features can each be predictive but highly correlated with one another.
- A feature with low individual correlation may still add value when combined with another feature.

For this reason, correlation is often useful for exploration and removing redundant inputs, rather than as the only feature-selection method.

### 6.2 Mutual information

Mutual information measures how much information a feature provides about the target. Unlike Pearson correlation, it can capture some nonlinear relationships.

```
from sklearn.feature_selection import mutual_info_classifmi_scores = mutual_info_classif(    X_train,    y_train,    random_state=42,)mi_results = pd.Series(    mi_scores,    index=X_train.columns,).sort_values(ascending=False)print(mi_results)
```

Select the top 10 features:

```
top_10_mi_features = mi_results.head(10).index.tolist()X_train_mi = X_train[top_10_mi_features]X_test_mi = X_test[top_10_mi_features]print("Selected features:", top_10_mi_features)print("Reduced training shape:", X_train_mi.shape)
```

This code selects features based on the training set and applies the same columns to the test set.

Limitation: mutual information scores are estimates and can be noisy, particularly with limited data. Selecting the top 10 is a demonstration, not a universal rule. Compare several feature counts using cross-validation.

### 6.3 Variance threshold

A feature with no variation cannot distinguish between observations. A variance threshold can remove constant or nearly constant features.

```
from sklearn.feature_selection import VarianceThresholdvariance_selector = VarianceThreshold(threshold=0.0)X_train_var = variance_selector.fit_transform(X_train)X_test_var = variance_selector.transform(X_test)selected_columns = X_train.columns[    variance_selector.get_support()].tolist()print("Selected columns:", selected_columns)print("Before:", X_train.shape)print("After:", X_train_var.shape)
```

Notice the pattern:

- `fit_transform()` on training data learns which features to retain and transforms the training set.
- `transform()` applies the same learned selection to test data.

Variance thresholding does not use the target, so it cannot tell whether a varying feature is predictive. It is best for removing constant or nearly constant columns, not for selecting all the most useful features.

### When should you use filter methods?

Use them when you need a fast initial reduction of a very wide dataset. For important business models, validate the selected features with cross-validation and a downstream model rather than relying only on statistical scores.

## 7. Method 2 — Wrapper methods: Recursive Feature Elimination (RFE)

Wrapper methods evaluate feature subsets by repeatedly fitting a model. They can capture model-specific usefulness, but they cost more computation than filter methods.

RFE works as follows:

1. Train a model using the current features.
2. Rank the features according to the model's importance measure.
3. Remove the least important feature or features.
4. Repeat until the requested number remains.

### RFE with logistic regression

```
from sklearn.feature_selection import RFEfrom sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerrfe_pipeline = Pipeline([    ("scaler", StandardScaler()),    (        "selector",        RFE(            estimator=LogisticRegression(max_iter=2000),            n_features_to_select=10,            step=1,        ),    ),    (        "classifier",        LogisticRegression(max_iter=2000),    ),])rfe_pipeline.fit(X_train, y_train)selector = rfe_pipeline.named_steps["selector"]rfe_features = X_train.columns[selector.support_].tolist()print("RFE-selected features:")print(rfe_features)
```

Why use a pipeline?

The scaler and RFE selector are fitted only on the training data passed to `fit()`. When you use the pipeline to predict on new data, it applies the learned scaling and selected columns in the same order.

This prevents you from accidentally using test-set information to select features.

### How to choose the number of features

Don't assume that 10 is always best. Compare different feature counts through cross-validation, ideally with selection and scaling repeated within each fold. The best count depends on the data, the model, and the performance metric you care about.

## 8. Method 3 — Embedded selection with L1 regularization

L1 regularization is a popular embedded feature-selection technique.

For logistic regression, L1 regularization adds a penalty on the absolute values of the model coefficients:

\\[ \text{Objective}=\text{LogLoss}+\lambda\sum\_{j=1}^{p}|\beta_j| \\]

Here, \\(\beta_j\\) is the coefficient for feature \\(j\\), and \\(\lambda\\) controls the strength of regularization. In scikit-learn, a smaller `C` means stronger regularization.

L1 regularization can force some coefficients to exactly zero. Those features then do not contribute directly to the fitted linear model.

### Implementation

```
from sklearn.feature_selection import SelectFromModelfrom sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerl1_pipeline = Pipeline([    ("scaler", StandardScaler()),    (        "selector",        SelectFromModel(            LogisticRegression(                penalty="l1",                solver="liblinear",                C=0.1,                max_iter=2000,                random_state=42,            ),            threshold=1e-8,        ),    ),    (        "classifier",        LogisticRegression(            max_iter=2000,            random_state=42,        ),    ),])l1_pipeline.fit(X_train, y_train)l1_selector = l1_pipeline.named_steps["selector"]l1_features = X_train.columns[    l1_selector.get_support()].tolist()
```

Why scale before L1?

Regularization penalizes coefficients, and features on different numeric scales can produce coefficients with different magnitudes for reasons unrelated to their predictive usefulness. Scaling makes the regularization behave more comparably across numeric features.

### What are the limitations?

- Correlated features may compete with each other, so one can be selected while another is removed.
- Strong regularization may discard useful information.
- The chosen subset can change across training samples.
- A zero coefficient means the feature was not used directly by this fitted linear model; it does not prove the feature has no relationship with the target.

L1 is especially useful when you want a relatively compact, interpretable linear model.

## 9. Method 4 — Tree-based feature selection

Tree-based models such as Random Forest and Gradient Boosting can estimate feature importance from how they use features to make predictions.

For example, a Random Forest can measure impurity-based importance: features that contribute to reducing impurity across tree splits tend to receive higher scores.

### Code: Random Forest feature importance

```
from sklearn.ensemble import RandomForestClassifierforest = RandomForestClassifier(    n_estimators=300,    min_samples_leaf=3,    random_state=42,    n_jobs=-1,)forest.fit(X_train, y_train)importance = pd.Series(    forest.feature_importances_,    index=X_train.columns,).sort_values(ascending=False)print("Feature importance:")print(importance)selected_tree_features = importance.head(10).index.tolist()print("\nTop 10 features:")print(selected_tree_features)
```

You can then train another model using the selected features:

```
selected_forest = RandomForestClassifier(    n_estimators=300,    min_samples_leaf=3,    random_state=42,    n_jobs=-1,)selected_forest.fit(    X_train[selected_tree_features],    y_train,)predictions = selected_forest.predict(    X_test[selected_tree_features])print("Test accuracy:",      (predictions == y_test).mean())
```

This example demonstrates the mechanics, but it does not yet provide an unbiased comparison of the original and reduced feature sets. The test set should ideally be used only for final evaluation after feature-selection choices have been settled using validation data or cross-validation.

### Important limitation: feature importance is not causality

Tree-based importance can be misleading when:

- Features are strongly correlated.
- Features have many possible split points or categories.
- Importance is spread across substitute features.
- The dataset is small or noisy.

For a more reliable model-agnostic check, use permutation importance on validation data. It measures how much a chosen evaluation score worsens when a feature's values are shuffled.

```
from sklearn.inspection import permutation_importancefrom sklearn.metrics import average_precision_scoreresult = permutation_importance(    forest,    X_test,    y_test,    scoring="average_precision",    n_repeats=10,    random_state=42,    n_jobs=-1,)permutation_scores = pd.Series(    result.importances_mean,    index=X_test.columns,).sort_values(ascending=False)print(permutation_scores)
```

Use this code for exploratory analysis only if you're still making model or feature-selection decisions: in that case, calculate permutation importance on a validation set rather than the final test set. Repeatedly inspecting test-set importance and modifying the model can lead to test-set overfitting.

## 10. Production-level feature selection using cross-validation

So far, we have implemented individual methods. In a real ML project, a key requirement is to evaluate the entire workflow, including feature selection, inside cross-validation.

Why? Suppose you select the top 10 features using the whole training dataset, then run cross-validation on those same selected features. Each validation fold has already influenced the selection. This can produce an overly optimistic score.

Instead, put the selector inside the pipeline so it is refitted separately in every training fold.

### Complete example: mutual information + logistic regression

```
import numpy as npfrom sklearn.feature_selection import SelectKBest, mutual_info_classiffrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import StratifiedKFold, cross_validatefrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerpipeline = Pipeline([    (        "selector",        SelectKBest(            score_func=mutual_info_classif,            k=10,        ),    ),    ("scaler", StandardScaler()),    (        "model",        LogisticRegression(            max_iter=2000,            random_state=42,        ),    ),])cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42,)scores = cross_validate(    pipeline,    X_train,    y_train,
```

This pipeline does the following within every cross-validation fold:

1. Estimates feature relevance using only that fold's training portion.
2. Selects the top 10 features.
3. Scales the selected features.
4. Trains logistic regression.
5. Evaluates the model on the fold's held-out portion.

The `+/-` value is the standard deviation across folds, not a formal confidence interval.

One subtle point: `SelectKBest` does not guarantee that the same features will be selected in every fold. That is useful information. If the selected features vary substantially, the feature set may be unstable, or several features may carry similar information.

### Should you tune `k`?

Yes. Compare values such as 5, 10, 15, and 20 with cross-validation. If you also tune model hyperparameters, use a disciplined model-selection procedure, such as a grid search with the feature selector inside the pipeline. For an unbiased final performance estimate, use a held-out test set or nested cross-validation.

## 11. Common mistakes in feature selection

| Mistake                                        | Why it is a problem                            | Better approach                                           |
| ---------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Selecting features before splitting            | Leaks information from held-out data           | Split first; fit selection on training data               |
| Selecting features based on test scores        | Overfits the final evaluation set              | Use validation data or cross-validation                   |
| Using only correlation                         | Misses some nonlinear and interaction effects  | Compare multiple methods and downstream performance       |
| Keeping only the top features from one model   | Importance is model-dependent                  | Validate feature subsets with the intended model          |
| Removing all correlated features automatically | Some correlated features may have unique value | Check redundancy and incremental predictive value         |
| Using post-outcome data                        | Creates target leakage                         | Only use features available at prediction time            |
| Optimizing accuracy on rare-event data         | Can hide poor minority-class detection         | Use suitable metrics such as average precision and recall |

For financial applications, target leakage is especially important. For example, a model predicting whether a customer will default must not use a collection date or a recovery status recorded after the default event. Such variables can make the model appear excellent during testing while being unusable in real deployment.

## 12. How do you decide which method to use?

For a small or medium tabular dataset

Start with a baseline model, then compare L1 regularization, mutual information, and tree-based methods.

For hundreds or thousands of columns

Consider inexpensive filter methods first, followed by model-based selection and cross-validation.

For highly interpretable models

Consider L1-regularized logistic regression, domain-based feature selection, and stability checks.

For nonlinear relationships

Compare mutual information and tree-based methods, then evaluate performance on unseen data.

For production financial systems

Include time-aware validation where appropriate, ensure features are available at prediction time, and monitor feature stability and drift.

## 13. Senior ML engineer interview answer

If an interviewer asks, “How do you perform feature selection in a production machine learning project?”, a strong answer is:

> I start with data quality checks, domain knowledge, and leakage analysis. I establish a baseline model and compare filter methods such as mutual information, wrapper methods such as RFE, and embedded methods such as L1 regularization or tree-based feature importance.
>
> I keep feature selection inside the training pipeline so it is refitted within each cross-validation fold. I compare candidate feature sets using metrics aligned with the business problem, then evaluate the selected workflow on an untouched test set.
>
> I also check redundancy, feature stability, training cost, and interpretability. In production, I ensure all features are available at prediction time and monitor their distributions and predictive behavior over time. I do not assume that fewer features always produce a better model; I choose the feature set that gives the best validated trade-off between performance, reliability, and operational complexity.

The key takeaway: Feature selection is not simply deleting columns with low importance. It is a disciplined process of finding a reliable, useful set of inputs while avoiding leakage and verifying that the resulting model generalizes to unseen data.