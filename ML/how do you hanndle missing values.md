# How to Handle Missing Values in Machine Learning — With Python Code

Handling missing values is one of the most important parts of building a reliable machine learning pipeline. In real-world datasets, some customers may not provide their income, some transactions may lack location information, and some financial records may have incomplete fields.

As an ML engineer, you should understand not only how to fill missing values, but also why a value is missing, which imputation method is appropriate, and how to prevent data leakage in production.

We'll learn this step by step using Python, pandas, and scikit-learn.

## 1. What are missing values?

Consider this customer dataset:

| Customer | Age     | Income (₹) | Credit score | Default |
| -------- | ------- | ---------- | ------------ | ------- |
| A        | 25      | 500,000    | 720          | No      |
| B        | 32      | Missing    | 680          | No      |
| C        | Missing | 800,000    | 750          | No      |
| D        | 45      | 600,000    | Missing      | Yes     |
| E        | 38      | 450,000    | 610          | Yes     |

The missing values may be represented as:

- `NaN` in pandas and NumPy.
- `None` in Python objects.
- `NULL` in databases.
- Empty strings or special placeholders such as `"unknown"` in raw files.

These representations are not always treated the same way. For example, an empty string is not automatically considered missing by every pandas operation.

Our goal is to prepare reliable features for a machine learning algorithm without introducing misleading information.

## 2. First, identify missing values

Install the libraries:

```
python -m pip install pandas numpy scikit-learn
```

Create a file named `missing_values_basics.py`.

```
import pandas as pdimport numpy as npdata = pd.DataFrame({    "age": [25, 32, np.nan, 45, 38],    "income": [500000, np.nan, 800000, 600000, 450000],    "credit_score": [720, 680, 750, np.nan, 610],    "default": [0, 0, 0, 1, 1],})print("Dataset:")print(data)# Count missing values in every column.print("\nMissing values per column:")print(data.isnull().sum())# Percentage of missing values in every column.missing_percentage = data.isnull().mean() * 100print("\nMissing percentage:")print(missing_percentage.round(2))# Total missing cells.print("\nTotal missing cells:", data.isnull().sum().sum())
```

### How does this work?

- `data.isnull()` returns `True` for missing cells.
- `data.isnull().sum()` counts missing values per column.
- `data.isnull().mean() * 100` calculates the percentage of missing entries per column.
- `data.isnull().sum().sum()` counts all missing cells.

This is the first step in a production data-quality pipeline. Before imputing values, determine whether missingness is expected, a data collection problem, or a signal in itself.

## 3. Understand why values are missing

Before choosing an imputation technique, investigate why the data is missing. This affects which methods are appropriate.

There are three commonly discussed missingness mechanisms.

1\. MCAR — Missing Completely At Random

The missingness is unrelated to observed or unobserved data.

Example: a random system failure causes some income values to disappear.

Approach: Simple imputation may work well, although deleting rows can still waste valuable data.

2\. MAR — Missing At Random

Missingness depends on other observed variables.

Example: customers in a particular region are less likely to report income, and region is recorded.

Approach: Use observed features to guide imputation where justified, potentially with multivariate imputation.

3\. MNAR — Missing Not At Random

Missingness depends on the missing value itself or other unobserved information, even after accounting for observed data.

Example: people with very high incomes may be less likely to disclose their income.

Approach: Investigate the data collection process, retain missingness indicators where appropriate, and consider sensitivity analysis. Ordinary imputation alone cannot reliably solve this problem.

These categories are assumptions about the data-generating process, not labels that can always be determined conclusively from the dataset.

## 4. Method 1 — Remove rows or columns

The simplest approach is to delete records or features that contain missing values.

### A. Remove rows

```
import pandas as pdimport numpy as npdata = pd.DataFrame({    "age": [25, 32, np.nan, 45, 38],    "income": [500000, np.nan, 800000, 600000, 450000],    "credit_score": [720, 680, 750, np.nan, 610],})clean_data = data.dropna()print(clean_data)
```

`dropna()` removes every row that contains at least one missing value by default.

Advantages

- Easy to implement.
- Does not invent replacement values.
- Can be reasonable when only a few rows are missing and the missingness mechanism justifies deletion.

Disadvantages

- Discards potentially useful records.
- Can introduce bias if missingness is systematic.
- Can remove a large proportion of the dataset when multiple columns are incomplete.

### B. Remove columns

```
clean_data = data.dropna(axis=1, thresh=4)
```

Here, `thresh=4` means that a column must contain at least four non-missing values to be retained. In this five-row example, columns with fewer than four observed values are removed.

Do not delete a feature merely because its missing percentage is high. A rare but important feature may still be valuable, depending on the business problem.

## 5. Method 2 — Mean imputation

Mean imputation replaces a missing numerical value with the average of the observed values in that feature.

For example:

\\[ \text{Mean}=\frac{\sum\_{i=1}^{n}x_i}{n} \\]

Suppose ages are `20, 30, 40, missing`.

The mean is 30, so the missing age is replaced with 30.

### Complete code

```
import pandas as pdimport numpy as npfrom sklearn.impute import SimpleImputerdata = pd.DataFrame({    "age": [20, 30, 40, np.nan, 50],    "income": [30000, 40000, np.nan, 60000, 70000],})imputer = SimpleImputer(strategy="mean")imputed_array = imputer.fit_transform(data)imputed_data = pd.DataFrame(    imputed_array,    columns=data.columns,)print(imputed_data)print("Learned means:", imputer.statistics_)
```

### When should you use mean imputation?

Mean imputation can be useful when numerical features are reasonably well behaved and the mean is a sensible representative value.

However, it has limitations:

- It is sensitive to outliers.
- It reduces variability in the imputed feature.
- It can distort relationships between variables.
- It can weaken statistical inference if used without accounting for imputation uncertainty.

For income data with extreme values, the mean may be a poor choice.

## 6. Method 3 — Median imputation

Median imputation replaces missing numerical values with the median of observed values.

For example:

`20, 30, 40, 1000, missing`

The median of the observed values is 40, making it less affected by the extreme value of 1000 than the mean would be.

### Complete code

```
import pandas as pdimport numpy as npfrom sklearn.impute import SimpleImputerdata = pd.DataFrame({    "age": [25, 30, 35, np.nan, 40],    "income": [30000, 40000, 50000, np.nan, 2000000],})imputer = SimpleImputer(strategy="median")result = pd.DataFrame(    imputer.fit_transform(data),    columns=data.columns,)print("Median values:", imputer.statistics_)print(result)
```

In practice, median imputation is a strong baseline for skewed numerical features, such as income, transaction amounts, and account balances.

But median imputation is not automatically best for every feature. The correct method depends on the distribution, missingness mechanism, and downstream task.

## 7. Method 4 — Mode imputation for categorical features

Categorical features include values such as:

- Employment type: `Salaried`, `Self-employed`.
- Account type: `Savings`, `Current`.
- Region: `North`, `South`, `East`, `West`.

You cannot meaningfully calculate the arithmetic mean of these categories. A common baseline is to replace missing categories with the most frequent category, known as the mode.

```
import pandas as pdimport numpy as npfrom sklearn.impute import SimpleImputerdata = pd.DataFrame({    "employment_type": [        "Salaried",        "Self-employed",        "Salaried",        np.nan,        "Salaried",    ],    "account_type": [        "Savings",        "Current",        np.nan,        "Savings",        "Savings",    ],})imputer = SimpleImputer(strategy="most_frequent")result = pd.DataFrame(    imputer.fit_transform(data),    columns=data.columns,)print("Most frequent values:", imputer.statistics_)print(result)
```

### An alternative: use an explicit category

Sometimes missingness itself is meaningful. For example, a missing employment type might indicate that the customer did not disclose it.

You can preserve that information with a constant category:

```
imputer = SimpleImputer(    strategy="constant",    fill_value="Unknown",)
```

This avoids pretending that an unknown value is actually the most common category.

For categorical models, you may then encode the resulting categories using one-hot encoding or another suitable encoding strategy.

## 8. Method 5 — Missing-value indicators

This is an important technique for production ML.

Sometimes, the fact that a value is missing is itself predictive.

For example, in a financial dataset, customers who do not provide certain information may behave differently from customers who do. The missingness may carry information even after you impute the numerical feature.

Instead of only filling missing values, create an additional binary feature:

- `1` = original value was missing.
- `0` = original value was available.

### Complete code

```
import pandas as pdimport numpy as npdata = pd.DataFrame({    "age": [25, 30, np.nan, 45, 50],    "income": [30000, np.nan, 50000, 60000, 70000],})for column in data.columns:    data[f"{column}_was_missing"] = (        data[column].isna().astype(int)    )# Impute the original numeric columns.for column in ["age", "income"]:    data[column] = data[column].fillna(        data[column].median()    )print(data)
```

The output includes both the imputed features and the indicators.

| age | income | age_was_missing | income_was_missing |
| --- | ------ | --------------- | ------------------ |
| 25  | 30000  | 0               | 0                  |
| 30  | 50000  | 0               | 1                  |
| 35  | 50000  | 1               | 1                  |
| 45  | 60000  | 0               | 0                  |
| 50  | 70000  | 0               | 0                  |

The table illustrates the format of the resulting data; in this particular code, the median of the original observed ages is 37.5, so the third row's imputed age will be 37.5 rather than 35.

The indicator can help a model distinguish an imputed value from a value that was actually observed. However, use it only when it is appropriate and stable for the business context; missingness can also reflect data pipeline failures.

## 9. Method 6 — KNN imputation

Mean and median imputation treat each feature independently. K-Nearest Neighbors (KNN) imputation uses similarities between rows to estimate missing values.

Suppose two customers have similar ages, account balances, and spending behavior. If one customer's income is missing, KNN can use income values from similar customers to estimate it.

### How it works

1. Identify records with known values for the feature being imputed.
2. Find similar records based on the other available features.
3. Select the nearest neighbors.
4. Estimate the missing value using their known values.
5. Repeat for the missing entries.

### Complete code

```
import pandas as pdimport numpy as npfrom sklearn.impute import KNNImputerfrom sklearn.preprocessing import StandardScalerdata = pd.DataFrame({    "age": [25, 30, 35, 40, 45, 50],    "income": [30000, 35000, np.nan, 50000, 55000, 60000],    "credit_score": [650, 670, 690, np.nan, 730, 750],})# Scale the features before calculating distances.scaler = StandardScaler()scaled_data = scaler.fit_transform(data)# Estimate missing values using the nearest neighbors.imputer = KNNImputer(n_neighbors=3, weights="distance")imputed_scaled = imputer.fit_transform(scaled_data)# Return the results to the original feature scales.imputed_values = scaler.inverse_transform(imputed_scaled)result = pd.DataFrame(    imputed_values,    columns=data.columns,)print(result.round(2))
```

### Why scale the data?

KNN relies on distances. Income may range from 30,000 to 60,000, whereas age may range from 25 to 50. Without scaling, income could dominate the distance calculation simply because its numbers are larger.

`StandardScaler` puts the features on comparable scales.

Important: In a real ML pipeline, fit the scaler and KNN imputer on training data only. The code above demonstrates the algorithm on one dataset, but fitting on the entire dataset before splitting would cause data leakage in a predictive evaluation.

KNN imputation also has drawbacks:

- It can be expensive on large datasets.
- It can struggle with high-dimensional features.
- Results depend on feature scaling and the chosen number of neighbors.
- It may be unsuitable for mixed numerical and categorical data without additional design.

## 10. Method 7 — Iterative imputation

Iterative imputation is a more advanced approach.

Rather than filling every missing value with one fixed statistic, it repeatedly models each feature with missing values using other features.

For example:

- Predict missing income from age, credit score, and account activity.
- Predict missing credit score from age, income, and other available features.
- Repeat the estimation process for several rounds.

Scikit-learn provides `IterativeImputer`, which is experimental and must be explicitly enabled.

### Complete code

```
import numpy as npimport pandas as pdfrom sklearn.experimental import enable_iterative_imputer  # noqa: F401from sklearn.impute import IterativeImputerfrom sklearn.linear_model import BayesianRidgedata = pd.DataFrame({    "age": [25, 30, np.nan, 40, 45, 50, 55],    "income": [30000, 35000, 40000, np.nan, 50000, 55000, 60000],    "credit_score": [650, 670, 690, 700, np.nan, 730, 750],})imputer = IterativeImputer(    estimator=BayesianRidge(),    max_iter=10,    random_state=42,)imputed = imputer.fit_transform(data)result = pd.DataFrame(    imputed,    columns=data.columns,)print(result.round(2))
```

This is an example of model-based imputation. The imputed values are estimates, not recovered ground truth.

Iterative imputation can be useful when features have meaningful relationships, but it can introduce bias if its assumptions are inappropriate. If your goal is statistical inference, ordinary single imputation may understate uncertainty; multiple imputation methods may be more appropriate.

## 11. Compare the missing-value strategies

| Method               | Best suited for                                          | Main limitation                            |
| -------------------- | -------------------------------------------------------- | ------------------------------------------ |
| Drop rows            | Very few incomplete records, with justified deletion     | Information loss and potential bias        |
| Mean                 | Roughly symmetric numerical features                     | Sensitive to outliers                      |
| Median               | Skewed numerical features                                | Can weaken feature relationships           |
| Mode                 | Categorical features                                     | Dominant category may be misleading        |
| Constant / `Unknown` | Meaningful categorical missingness                       | Adds a category that may be sparse         |
| Missing indicator    | Missingness may itself be informative                    | May learn unstable collection patterns     |
| KNN                  | Similar records can inform missing values                | Computational cost and scaling sensitivity |
| Iterative imputation | Relationships between numerical features are informative | Model assumptions and added complexity     |

There is no universally best method. A senior ML engineer compares candidate strategies using cross-validation and selects one based on held-out predictive performance, data quality, and operational constraints.

## 12. Most important: prevent data leakage

This is one of the most common mistakes in ML projects.

Consider this incorrect approach:

```
# Incorrect for a predictive evaluation:X_imputed = SimpleImputer(strategy="median").fit_transform(X)X_train, X_test, y_train, y_test = train_test_split(    X_imputed, y, test_size=0.2, random_state=42)
```

The imputer has already calculated medians using the entire dataset, including the future test set.

Although it does not use the target labels, it has allowed information from the test distribution to influence preprocessing.

Correct approach: Split the data first, then fit the imputer only on training data. Better still, put preprocessing inside a scikit-learn `Pipeline`.

## 13. Production-style example — Build a leakage-safe ML pipeline

Let's combine numerical imputation, categorical imputation, encoding, and supervised classification in a single pipeline.

This is a reusable pattern for real ML projects.

```
import numpy as npimport pandas as pdfrom sklearn.compose import ColumnTransformerfrom sklearn.impute import SimpleImputerfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import OneHotEncoder, StandardScalerfrom sklearn.ensemble import HistGradientBoostingClassifierfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import classification_report, roc_auc_score# 1. Example dataset with missing numerical and categorical values.data = pd.DataFrame({    "age": [25, 32, np.nan, 45, 38, 29, 51, 42, 35, 48],    "income": [        50000, np.nan, 80000, 60000, 45000,        52000, 90000, np.nan, 48000, 75000    ],    "employment_type": [        "Salaried", "Self-employed", "Salaried",        np.nan, "Salaried", "Self-employed",        "Salaried", "Salaried", np.nan, "Self-employed"    ],    "default": [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],})# 2. Separate features and target.X = data.drop(columns=["default"])y = data["default"]numeric_features = ["age", "income"]categorical_features = ["employment_type"]# 3. Numerical preprocessing.numeric_pipeline = Pipeline([
```

This is an architectural example, not a model ready for financial decisions. Ten synthetic records are far too few for meaningful evaluation, and this particular split can be unstable. Use a substantially larger representative dataset for real training.

### Why this pipeline is better

- The imputer learns statistics from training data only.
- Missing numerical values are replaced with training-set medians.
- Numerical missingness indicators are added automatically.
- Missing categories are represented as `Unknown`.
- One-hot encoding handles categorical features.
- The entire pipeline can be cross-validated and saved as one artifact.
- New data receives the same transformations used during training.

One subtle point: `HistGradientBoostingClassifier` can natively handle missing numerical values. In this pipeline, we deliberately impute them to demonstrate explicit, controlled preprocessing. In other projects, native missing-value handling may be preferable and should be evaluated against explicit imputation.

Also, use time-based splits rather than random splits when the production task requires predicting future outcomes from historical records.

## 14. How should you choose an imputation strategy?

Use this decision process as your starting point.

Investigate missingness

Data quality, missing percentage, collection process and feature meaning

Identify feature type

Numerical, categorical, time series or structured records

Numerical

Median, mean, KNN, iterative or native model support

Categorical

Mode, `Unknown`, or a domain-specific missing category

Validate with a pipeline

Compare strategies with cross-validation; keep preprocessing inside the pipeline.

Deploy and monitor

Version preprocessing, validate input quality, track missingness drift and model performance.

## 15. Senior ML engineer interview questions

Make sure you can answer these:

1. Why is median imputation often preferred over mean imputation for income?
2. What is the difference between MCAR, MAR, and MNAR?
3. Why must an imputer be fitted only on training data?
4. When would you use KNN imputation instead of a simple imputer?
5. What is a missingness indicator, and when could it introduce risk?
6. How would you handle a categorical feature that contains a previously unseen category in production?
7. How would you handle missing values in time-series data without leaking future information?
8. When might you let a tree-based model handle missing values natively rather than imputing them?
9. How would you evaluate whether imputation improves predictive performance?
10. How would you detect a production data pipeline that suddenly starts producing many more missing values?

The key lesson: Missing-value handling is not simply replacing `NaN` with a number. It is a data-quality and modeling decision. In a production ML system, understand why the data is missing, choose an appropriate strategy, fit preprocessing without leakage, and validate the complete pipeline against a realistic baseline.