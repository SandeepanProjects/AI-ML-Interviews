# How to Handle Outliers in Machine Learning — From Fundamentals to Production Code

Outlier handling is an important part of data preprocessing in machine learning. If you are building models for financial risk, fraud detection, customer analytics, or revenue prediction, you need to understand how to identify unusual values and decide whether to keep, transform, cap, or remove them.

The most important principle is:

An outlier is an unusual observation, not necessarily an incorrect observation.

For example, a transaction of ₹10 lakh might be an error in one dataset, a legitimate high-value purchase in another, or exactly the kind of event a fraud detection system needs to identify.

We'll cover the techniques step by step, using Python, pandas, NumPy, and scikit-learn.

## 1. What is an outlier?

An outlier is an observation that differs substantially from the overall pattern of the data.

Consider this customer income dataset:

```
income = [30000, 32000, 35000, 31000, 34000, 33000, 2000000]
```

Most incomes are between ₹30,000 and ₹35,000, but one value is ₹20 lakh.

That value might be:

- A data entry error.
- A legitimate high-income customer.
- A value recorded in a different currency.
- A special case that needs separate treatment.

You should investigate the observation before deciding what to do with it.

### Types of outliers

[Data Quality & Outliers: Complete Guide to Measurement Error, Missing Data & Detection Methods - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/Qu7bDNIPO_inmLGANsfSwJTfCOggXgF6AZQnumtf-ugjwCeEE2VSM-Z9Lgkd7Alnsx7xtsWoPSyKyyEMnZdOaAtpoBiaP4L6t1yCHCb3b5PPGigG9dCw4NGIW2tT9BLLJG9_dB7LQtDaMPHUaU2J05pEJoxFvN-OBDyfC77U-nk?purpose=inline)

1\. Statistical outliers

Values unusually far from the distribution, identified using methods such as IQR, z-scores, or robust statistical techniques.

[Top Tools for Data Exploration | Encord](https://images.openai.com/static-rsc-4/1vcqcgpNA1FSgJsTBQalSrQrgRY2TJzYTwt79A8cWZzmmhZwgczbIMq2uk_kSq6C0VoOyH0CRU7gfIXn70Kkj29yG_8wHcFNGvKmPc6DShaRH2GsgEotevXGQJkaSo957310gmmXrMbLzRCsHUvHb4wXyjsZCm9MB5sdnD1QRXo?purpose=inline)

2\. Contextual outliers

Values that are unusual in a particular context. A ₹50,000 purchase might be normal for a business account but unusual for a small personal account.

[Isolation Forest: Complete Guide to Unsupervised Anomaly Detection with Random Trees & Path Length Analysis - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/pW-1OWsG474pSKVnIX24LJsBss7UgVDlEBe7wqzkNfTuvRw_qGOcuudPCoI5D6igZqkIKTbKGt3muBGd2hxVOEEjgNn9vaYO8Bp3Unt1aIhnLaJmT0NfdfV1YSymfAIgjB-CxG9I47WCdj8I_gq5QcXN5vzIJWop7MRz6h6Ut30?purpose=inline)

3\. Multivariate outliers

Observations that look normal in individual features but unusual when multiple features are considered together.

For example, a customer's age and income might each be plausible, but their combination with transaction frequency and account history might be unusual.

## 2. First, create a dataset containing outliers

Let's use a synthetic income dataset.

```
import numpy as npimport pandas as pddata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ],    "age": [        25, 30, 28, 35,        40, 32, 45, 29,        31, 38, 33, 42,    ],})print(data.describe())print("\nMaximum income:", data["income"].max())print("Median income:", data["income"].median())
```

The median income is around ₹33,000, while the maximum is ₹20 lakh.

This extreme observation can distort some statistics and some machine learning models.

For example, the mean is much more affected by the ₹20 lakh value than the median is.

But remember: this alone does not establish that the value is invalid.

## 3. Method 1 — Detect outliers using the IQR method

The Interquartile Range (IQR) method is one of the most common ways to identify statistical outliers in numerical data.

First, understand the quartiles:

- Q1: The 25th percentile.
- Q2: The median, or 50th percentile.
- Q3: The 75th percentile.

The IQR is the distance between Q3 and Q1.

\\[ IQR=Q3-Q1 \\]

The traditional Tukey rule flags observations below or above these boundaries:

\\[ \text{Lower}=Q1-1.5(IQR) \\]

\\[ \text{Upper}=Q3+1.5(IQR) \\]

Values outside the boundaries are flagged as potential outliers. They are not automatically errors.

### Complete Python code

```
import pandas as pddata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})# 1. Calculate quartiles.q1 = data["income"].quantile(0.25)q3 = data["income"].quantile(0.75)# 2. Calculate IQR.iqr = q3 - q1# 3. Calculate outlier boundaries.lower_bound = q1 - 1.5 * iqrupper_bound = q3 + 1.5 * iqr# 4. Flag outliers.data["is_outlier"] = (    (data["income"] < lower_bound)    | (data["income"] > upper_bound))print("Q1:", q1)print("Q3:", q3)print("IQR:", iqr)print("Lower boundary:", lower_bound)print("Upper boundary:", upper_bound)print("\nPotential outliers:")print(data[data["is_outlier"]])
```

### What does this code do?

Suppose the calculated upper boundary is ₹36,000. An income of ₹20 lakh would be flagged because it exceeds that boundary.

The actual boundary depends on the dataset's quartiles, so you should use the code's output rather than assume a fixed threshold.

When to use IQR: It is a good starting point for skewed numerical features and exploratory data analysis. It does not assume a normal distribution, although the rule may flag many valid observations in some distributions.

## 4. Method 2 — Detect outliers using the Z-score

The Z-score measures how far a value is from the mean in standard deviation units.

\\[ z=\frac{x-\mu}{\sigma} \\]

Where:

- \\(x\\) is the observed value.
- \\(\mu\\) is the mean.
- \\(\sigma\\) is the standard deviation.

A common rule flags values with an absolute Z-score greater than 3.

### Complete Python code

```
import pandas as pdfrom scipy.stats import zscoredata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})data["z_score"] = zscore(    data["income"],    ddof=0,)data["is_outlier"] = data["z_score"].abs() > 3print(data)print("\nPotential outliers:")print(data[data["is_outlier"]])
```

Install SciPy if needed:

```
python -m pip install scipy
```

### IQR vs Z-score

| IQR                                | Z-score                                                        |
| ---------------------------------- | -------------------------------------------------------------- |
| Based on quartiles                 | Based on mean and standard deviation                           |
| More robust to extreme values      | Mean and standard deviation can be distorted by extreme values |
| Useful for skewed distributions    | Most interpretable for roughly normal distributions            |
| Uses a conventional 1.5 × IQR rule | Often uses an absolute Z-score threshold of 3                  |

One subtle issue: a single extreme value can inflate the standard deviation and make its own Z-score smaller than expected. With small samples or multiple outliers, neither method should be treated as definitive.

## 5. Method 3 — Remove outliers

If you have confirmed that observations are invalid, removing them may be appropriate.

For example, suppose age must be between 18 and 120 because the dataset contains adult customers. A record with age 250 could be a data entry error.

### A. Remove using an IQR rule

```
import pandas as pddata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})q1 = data["income"].quantile(0.25)q3 = data["income"].quantile(0.75)iqr = q3 - q1lower = q1 - 1.5 * iqrupper = q3 + 1.5 * iqrclean_data = data[    data["income"].between(lower, upper)].copy()print("Original rows:", len(data))print("Remaining rows:", len(clean_data))print(clean_data)
```

### When is removal appropriate?

Removal can be reasonable when:

- The record is demonstrably corrupt or invalid.
- A unit-conversion or data-entry error has been confirmed.
- The observation falls outside a valid domain defined by the data source.
- A documented exclusion rule is justified for the task.

Avoid automatically deleting every statistically unusual value. In fraud detection, for example, rare extreme transactions may be the most valuable records in your dataset.

Also, if you remove training records, document the reason and evaluate whether the exclusion changes the population your model will serve.

## 6. Method 4 — Capping outliers (Winsorization)

Instead of removing unusual values, you can cap them at a specified lower or upper boundary.

For example, if the chosen upper limit is ₹50,000, any income above that limit becomes ₹50,000 in the transformed feature.

This preserves the record but reduces the influence of extreme values.

### Complete Python code using IQR boundaries

```
import pandas as pddata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})# Calculate limits from the available training data.q1 = data["income"].quantile(0.25)q3 = data["income"].quantile(0.75)iqr = q3 - q1lower = q1 - 1.5 * iqrupper = q3 + 1.5 * iqr# Preserve the original values.data["income_capped"] = data["income"].clip(    lower=lower,    upper=upper,)print(data)
```

`clip()` replaces values below the lower boundary with the lower boundary and values above the upper boundary with the upper boundary.

### Another common strategy: percentile capping

You might cap at the 1st and 99th percentiles instead:

```
lower = data["income"].quantile(0.01)upper = data["income"].quantile(0.99)data["income_capped"] = data["income"].clip(    lower=lower,    upper=upper,)
```

The percentile thresholds are examples, not universal recommendations. With a small dataset, the 1st and 99th percentiles may provide very little meaningful trimming.

Important: Capping changes the feature values. Keep the original value when auditability matters, and choose the thresholds using training data only. Do not silently cap a value if the extreme magnitude is itself an important business signal.

## 7. Method 5 — Apply a logarithmic transformation

Many real-world features are right-skewed. For example, income, transaction amounts, and customer spending may have a long tail of large values.

A logarithmic transformation compresses the scale of large values.

\\[ x\_{\text{transformed}}=\log(1+x) \\]

For non-negative values, `np.log1p(x)` calculates \\(\log(1+x)\\) accurately, including for values near zero.

### Complete Python code

```
import pandas as pdimport numpy as npdata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})data["income_log"] = np.log1p(data["income"])print(data)print("\nOriginal mean:", data["income"].mean())print("Log-transformed mean:", data["income_log"].mean())
```

The transformed value of ₹20 lakh is much closer to the other values on the log scale than on the original scale.

However, the outlier has not been removed. It is still the largest observation.

### When should you use log transformation?

- When a numerical feature is strongly right-skewed.
- When multiplicative relationships are plausible.
- When you want to reduce the influence of large magnitudes on models sensitive to scale or linear relationships.

Be careful with negative values: `np.log1p(x)` is only defined for \\(x>-1\\). For mixed positive and negative data, consider a transformation such as Yeo–Johnson instead.

Log transformation is often useful for linear or distance-based models. Tree-based models are generally less sensitive to feature scale, so transforming their inputs may or may not improve performance.

## 8. Method 6 — Use RobustScaler

Sometimes the problem is not that outliers are invalid, but that the model is sensitive to feature scale.

`StandardScaler` uses the mean and standard deviation. Extreme values can influence both.

`RobustScaler` instead centers numerical features using the median and scales them using the interquartile range.

\\[ x\_{\text{scaled}}= \frac{x-\operatorname{median}(X)} {Q3-Q1} \\]

### Complete Python code

```
import pandas as pdfrom sklearn.preprocessing import StandardScaler, RobustScalerdata = pd.DataFrame({    "income": [        30000, 32000, 35000, 31000,        34000, 33000, 36000, 31000,        32000, 35000, 33000, 2000000,    ]})standard_scaled = StandardScaler().fit_transform(data)robust_scaled = RobustScaler().fit_transform(data)data["standard_scaled"] = standard_scaled[:, 0]data["robust_scaled"] = robust_scaled[:, 0]print(data)
```

RobustScaler reduces the effect of extreme values on the estimated center and scale. It does not remove outliers; the transformed extreme value may still be far from the rest.

Use it when you need scaling for models such as KNN, SVM, or regularized linear models and your features contain extreme values.

## 9. Method 7 — Detect multivariate outliers with Isolation Forest

So far, we've looked at one feature at a time. But an observation can be unusual because of the combination of several features.

For example:

- An unusually large transaction with an unusual time and location.
- A customer with normal income but an unusual spending pattern.
- A server with individually plausible CPU and memory readings but an abnormal combination.

Isolation Forest is an unsupervised anomaly-detection algorithm. It repeatedly partitions the data and identifies observations that are isolated with relatively few splits.

### Complete Python code

```
import pandas as pdfrom sklearn.ensemble import IsolationForestdata = pd.DataFrame({    "transaction_amount": [        100, 120, 90, 110, 130, 105,        95, 115, 125, 1000,    ],    "transactions_per_day": [        2, 3, 2, 2, 3, 2,        1, 3, 2, 30,    ],})model = IsolationForest(    n_estimators=200,    contamination=0.1,    random_state=42,)# Fit the anomaly detector.model.fit(data)# 1 = inlier, -1 = anomaly.data["anomaly_label"] = model.predict(data)# Lower scores indicate more anomalous observations.data["anomaly_score"] = model.decision_function(data)print(data.sort_values("anomaly_score"))print("\nFlagged observations:")print(data[data["anomaly_label"] == -1])
```

### Understand the output

- `predict()` returns `-1` for observations flagged as anomalies and `1` for inliers.
- `decision_function()` returns anomaly-related scores; lower values are more anomalous.
- `contamination=0.1` configures the expected proportion of anomalies used to determine the threshold in this example. It is not a guarantee that 10% of future observations are actually anomalous.

Isolation Forest is useful for identifying suspicious records, but it does not determine whether those records are data errors, fraudulent activity, or legitimate rare events. That requires domain validation.

## 10. Which machine learning models are sensitive to outliers?

Outlier handling depends on the model you plan to train.

| Model               | Sensitivity to outliers                 | Practical consideration                                                                         |
| ------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Linear regression   | Often sensitive                         | Extreme feature values can have high leverage; extreme targets can influence squared-error loss |
| Logistic regression | Can be sensitive                        | Extreme features and influential observations may affect fitted coefficients                    |
| KNN                 | Often sensitive                         | Extreme values and feature scales can distort distances                                         |
| SVM                 | Depends on kernel and settings          | Outliers can affect the decision boundary, especially without appropriate regularization        |
| Decision tree       | Less sensitive to feature scale         | Extreme target values can still influence regression trees                                      |
| Random Forest       | Often robust to feature extremes        | Extreme targets and contaminated training records can still affect predictions                  |
| Gradient Boosting   | Depends on loss and tree structure      | Extreme target values can influence squared-error boosting                                      |
| Neural networks     | Often sensitive to poorly scaled inputs | Scaling, robust losses, and data-quality controls can help                                      |

This table describes general tendencies, not guarantees.

For example, tree-based models do not usually need StandardScaler or RobustScaler just to compare feature magnitudes, because their splits depend on thresholds rather than Euclidean distances. But that does not make them immune to bad records or extreme target values.

## 11. Production-style implementation: detect and cap without data leakage

Let's build a reusable scikit-learn pipeline that:

1. Splits the dataset into training and test sets.
2. Learns outlier thresholds from training data only.
3. Caps extreme numerical feature values.
4. Trains a regression model.
5. Evaluates performance on held-out data.

We will implement the capping logic as a transformer so it can be used inside a pipeline.

```
import numpy as npimport pandas as pdfrom sklearn.base import BaseEstimator, TransformerMixinfrom sklearn.datasets import make_regressionfrom sklearn.linear_model import Ridgefrom sklearn.model_selection import train_test_splitfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.metrics import mean_absolute_error, mean_squared_errorclass IQRCapper(BaseEstimator, TransformerMixin):    """Cap numeric features using training-set IQR boundaries."""    def __init__(self, factor=1.5):        self.factor = factor    def fit(self, X, y=None):        X = np.asarray(X, dtype=float)        self.q1_ = np.nanpercentile(X, 25, axis=0)        self.q3_ = np.nanpercentile(X, 75, axis=0)        iqr = self.q3_ - self.q1_        self.lower_ = self.q1_ - self.factor * iqr        self.upper_ = self.q3_ + self.factor * iqr        return self    def transform(self, X):        X = np.asarray(X, dtype=float)        # Preserve missing values; imputation is a separate step.
```

This example demonstrates the architecture, not a guarantee that IQR capping improves performance. The synthetic features are not actual income or spending measurements, and their target was generated before the artificial extremes were inserted.

The custom transformer also assumes numeric features. In a production implementation, handle missing values in a dedicated pipeline step, validate numeric types, and add explicit checks for infinities and invalid domain values.

Most importantly, do not fit outlier thresholds independently on the test data. The test data must be transformed using the thresholds learned from the training data.

## 12. How do you decide what to do with an outlier?

Use this practical decision framework.

Detect an unusual observation

Investigate the source and context

Check units, business rules, data collection, and related features.

Confirmed data error

Correct it from the source, or exclude it using a documented rule.

Valid extreme value

Preserve it unless there is a justified modeling reason to transform or cap it.

Skewed distribution

Evaluate a log or other suitable transformation.

Unusual feature combination

Investigate multivariate anomaly detection.

Compare models with realistic validation

Evaluate the untreated baseline against justified preprocessing alternatives.

## 13. Senior ML engineer interview questions

Here are six questions worth practicing:

1\. Should we always remove outliers before training?

2\. Why is the IQR method useful?

3\. What is the difference between capping and log transformation?

4\. Why are tree-based models often less sensitive to outliers?

5\. How do you prevent data leakage when handling outliers?

6\. How would you handle outliers in fraud detection?

## Final takeaway

For a production ML engineer, the right approach is not simply to detect and remove extreme values. It is to:

1. Understand the data and investigate unusual observations.
2. Choose an appropriate detection method, such as IQR, Z-score, or Isolation Forest.
3. Decide whether to correct, remove, cap, transform, or preserve the values.
4. Fit preprocessing only on training data.
5. Compare predictive performance using a realistic validation strategy.
6. Monitor outlier rates in production so that changes in data quality or customer behavior are detected.

My recommendation: Practice outlier handling alongside missing-value imputation in one end-to-end scikit-learn pipeline. This will teach you how preprocessing decisions interact with the model, evaluation, and deployment rather than treating each technique as an isolated step.