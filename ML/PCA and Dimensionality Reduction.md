# PCA and Dimensionality Reduction in Machine Learning — From Fundamentals to Production Code

PCA (Principal Component Analysis) is one of the most important dimensionality-reduction techniques in machine learning. It is used in data preprocessing, exploratory data analysis, visualization, compression, and some production ML pipelines.

I'll explain it step by step: the intuition, mathematics, Python implementation, how to choose the number of components, how to evaluate the results, and when PCA is or isn't appropriate in a production system.

## 1. What is dimensionality reduction?

In machine learning, a dimension usually refers to an input feature.

Suppose you're building a model to predict whether a customer will default on a loan. Your dataset contains 100 features:

- Annual income
- Credit score
- Outstanding debt
- Number of missed payments
- Monthly expenses
- Account balance
- And 94 more features

Your input matrix has 100 columns, so it has 100 feature dimensions.

Dimensionality reduction is the process of representing data using fewer dimensions while trying to preserve the information that matters for a particular purpose.

For example:

Original feature space

## 100 features

Income, debt, credit score, expenses, etc.

Apply dimensionality reduction

Reduced feature space

## 10 components

A compressed representation of the original features

The goal might be to reduce computation, remove redundant information, visualize data, or simplify a downstream model.

One important clarification: reducing 100 features to 10 does not automatically mean you retain 100% of the useful information. The quality of the reduced representation depends on the method and the task.

## 2. Why do we need dimensionality reduction?

Imagine you have 1 million customer records with 500 numeric features.

A high-dimensional dataset can create several challenges:

- Redundancy: income, spending, and account-balance features may be correlated.
- Computational cost: some algorithms become expensive when the number of features grows.
- Noise: irrelevant or noisy dimensions can make some learning tasks harder.
- Visualization: it's difficult to plot 500 dimensions directly.
- Distance calculations: in high-dimensional spaces, distances can become less informative for certain algorithms.

This last issue is related to the curse of dimensionality. As dimensionality increases, data can become sparse relative to the space, and some algorithms require much more data to learn reliable patterns.

Dimensionality reduction can help, but it isn't a universal fix. Tree-based models, for example, may work well with many original features and may not benefit from PCA at all.

## 3. What is PCA?

Principal Component Analysis (PCA) is a linear dimensionality-reduction technique that transforms the original numeric features into a smaller set of new, uncorrelated variables called principal components.

PCA finds directions in the data along which the observations vary the most.

Imagine plotting customers using two features:

- Monthly income
- Monthly spending

If high-income customers generally spend more, the points may form an elongated cloud.

PCA can identify the direction along which this cloud varies most. It then creates a new axis along that direction, called the first principal component.

[Principal Component Analysis — Statistical backgrounds for genetics trainees](https://images.openai.com/static-rsc-4/_9BVXykwlVSm3kSNaApUtED7Db-v8Lylr9PMYS5C1EWfIlWFRhjJLNnrumyFsP73DYASK_rKifra8UJTcEM3TshkBeI8GHRuI6O5jwPTySXWAQL4dW4WCjlIEFEoMh0nsUW8EPgGI3qSPBCWgPFZCsC6V1s5RHmqpOuG7DGZZao?purpose=inline)

[Principal Component Analysis, Part 2 — Algorithms for Data Science](https://images.openai.com/static-rsc-4/twljGsshDu0_O-rHqrCapHWTQhynXNLsjJpw5Nc6jVvM3ejUNavj7HwJOtv-ZbohR0f1a3gEpFmhnecmJRzPvyV8r6zGN-mYE1jAILY9vgSLassDjsyL_tVzDRXERgnCuqXUYsEfGbzSeISdP2uJwEIYU47mUD6E2I4DEwUBiJY?purpose=inline)

[Simulations - Probability & Machine Learning](https://images.openai.com/static-rsc-4/D0NLmjeFQmWIsttsIUS16CSnUPo7jj7b69wXWxnDFF07rKM0UjGRm0kvWopQT_cTuk9eM94xqZQiNK1cAKU0mRmgtD1Oe4VpDuNyQxU3Kl_xnnubdZcR3POI_zWJ3dx_0y7PQbcNdB-tJkiKruMXjmlJv_3-1hIGIYlLqqBsBa8?purpose=inline)

9

The first principal component, PC1, captures the direction of greatest variance. The second component, PC2, captures the greatest remaining variance subject to being perpendicular to PC1.

For two input features, PCA can produce two components. For 100 input features, it can produce up to 100 components, but we often retain only a smaller number.

### A simple example

Suppose these features are strongly correlated:

- `monthly_income`
- `annual_income`

Annual income is approximately monthly income multiplied by 12.

Both columns largely express the same underlying information. PCA can combine information from them into a new component instead of treating them as two independent directions.

PCA does not literally decide that one original column is important and the other is unimportant. It creates new combinations of the original columns.

That is the key difference between PCA and feature selection.

## 4. PCA vs. feature selection

This is an important distinction for ML interviews and real-world design.

| Feature selection                             | PCA                                                     |
| --------------------------------------------- | ------------------------------------------------------- |
| Keeps selected original columns               | Creates new combinations of columns                     |
| Example: keep income and credit score         | Example: create PC1 and PC2 from many features          |
| Original feature names remain                 | Components are usually less interpretable               |
| Can use the target label to select features   | Standard PCA is unsupervised; it doesn't use the target |
| Useful when business interpretability matters | Useful when compression or visualization matters        |

For example, feature selection might keep `income`, `credit_score`, and `debt_ratio`.

PCA might produce:

\\[ PC_1 = 0.5x_1+0.3x_2-0.4x_3+\cdots \\]

The actual coefficients are learned from the training data. PC1 is a mathematical combination, not a business variable with a directly meaningful name.

## 5. How PCA works mathematically

You don't need to calculate PCA manually in everyday ML development, but understanding the main steps helps you use it correctly.

### Step 1: Standardize the features

Suppose your dataset contains:

- Annual income: 1,200,000
- Credit score: 720
- Number of missed payments: 2

These variables have very different numeric scales. If PCA is applied directly, large-scale variables can dominate the variance calculation.

A common solution is standardization:

\\[ z=\frac{x-\mu}{\sigma} \\]

Where:

- \\(x\\) is the original value.
- \\(\mu\\) is the training-set mean.
- \\(\sigma\\) is the training-set standard deviation.
- \\(z\\) is the standardized value.

After standardization, each nonconstant feature has mean approximately zero and standard deviation approximately one on the training data.

### Step 2: Find the principal directions

PCA analyzes how features vary together, commonly through the covariance matrix of the standardized data.

It finds directions called eigenvectors, associated with eigenvalues.

- Eigenvector: a direction in feature space.
- Eigenvalue: the amount of variance along that direction.

Modern PCA implementations typically use numerical linear algebra methods such as singular value decomposition (SVD), rather than manually computing eigenvectors of the covariance matrix.

### Step 3: Rank components by explained variance

The first component captures the greatest possible variance. Each subsequent component captures the greatest remaining variance subject to being orthogonal to the previous components.

### Step 4: Project the data

The standardized data is projected onto the selected component directions.

If \\(X\\) is the standardized data matrix and \\(W_k\\) contains the first \\(k\\) principal directions, then:

\\[ Z=XW_k \\]

Here, \\(Z\\) is the reduced representation.

For example:

\\[ X:(5000,20)\quad\longrightarrow\quad Z:(5000,5) \\]

You still have 5,000 observations, but you now represent each one using five principal components instead of 20 original features.

## 6. Complete Python example: PCA from start to finish

Let's use the synthetic classification dataset again. This will demonstrate how to reduce 20 original features to fewer components.

### Step 1: Create the dataset and split it

```
import numpy as npimport pandas as pdfrom sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitX, y = make_classification(    n_samples=5000,    n_features=20,    n_informative=8,    n_redundant=6,    n_classes=2,    random_state=42,)feature_names = [f"feature_{i}" for i in range(X.shape[1])]X = pd.DataFrame(X, columns=feature_names)X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    stratify=y,    random_state=42,)print("Original training shape:", X_train.shape)print("Original test shape:", X_test.shape)
```

Expected shapes:

```
Original training shape: (4000, 20)
Original test shape: (1000, 20)
```

We split first so the PCA transformation is learned only from training data.

### Step 2: Standardize the data and apply PCA

```
from sklearn.preprocessing import StandardScalerfrom sklearn.decomposition import PCA# Fit the scaler on training data onlyscaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)X_test_scaled = scaler.transform(X_test)# Retain 5 principal componentspca = PCA(n_components=5)X_train_pca = pca.fit_transform(X_train_scaled)X_test_pca = pca.transform(X_test_scaled)print("Before PCA:", X_train_scaled.shape)print("After PCA:", X_train_pca.shape)print("Test shape after PCA:", X_test_pca.shape)
```

Expected output:

```
Before PCA: (4000, 20)
After PCA: (4000, 5)
Test shape after PCA: (1000, 5)
```

Why do we call `fit_transform()` only on training data?

`fit_transform()` learns the mean, scale, and principal directions from training observations. The test set is then transformed using those same learned values.

If you fit PCA on both training and test data, information from the test-set distribution can influence the transformation. This is a form of data leakage.

### Step 3: Calculate explained variance

The central question is: How much of the original variance is retained by those five components?

```
explained_variance = pca.explained_variance_ratio_for index, variance in enumerate(explained_variance, start=1):    print(f"PC{index}: {variance:.4f} ({variance * 100:.2f}%)")print(    "Total explained variance:",    round(explained_variance.sum() * 100, 2),    "%",)
```

`explained_variance_ratio_` reports the fraction of total training-data variance explained by each component.

For example, if the output were:

```
PC1: 0.25 (25.00%)
PC2: 0.18 (18.00%)
PC3: 0.12 (12.00%)
PC4: 0.09 (9.00%)
PC5: 0.07 (7.00%)
Total explained variance: 61.00 %
```

These numbers are illustrative; your actual output will depend on the generated data.

That hypothetical result means five components retain 61% of the standardized input variance. It does not mean they retain 61% of the predictive signal for the classification target.

### Step 4: Calculate cumulative explained variance

```
cumulative_variance = np.cumsum(    pca.explained_variance_ratio_)for index, variance in enumerate(cumulative_variance, start=1):    print(        f"First {index} components retain "        f"{variance * 100:.2f}% of variance"    )
```

Cumulative explained variance helps you understand the trade-off between compression and reconstruction quality.

## 7. How do you choose the number of principal components?

There are three common approaches.

### Approach A: Choose a fixed number

```
PCA(n_components=5)
```

Useful when you have a known computational or dimensionality constraint.

### Approach B: Retain a target percentage of variance

For example:

```
pca = PCA(n_components=0.95)X_train_pca = pca.fit_transform(X_train_scaled)X_test_pca = pca.transform(X_test_scaled)print("Original dimensions:", X_train_scaled.shape[1])print("Components selected:", pca.n_components_)print(    "Explained variance retained:",    pca.explained_variance_ratio_.sum(),)
```

Here, `0.95` asks PCA to retain enough components to explain at least 95% of the training data's variance, subject to the data and solver constraints.

This is a useful unsupervised heuristic, but 95% is not a universal optimum. A high-variance component may be irrelevant to your target, and a low-variance component may contain valuable predictive information.

### Approach C: Use a scree plot

A scree plot displays explained variance per component. An "elbow" in the curve can suggest a reasonable point beyond which additional components contribute relatively little variance.

```
import matplotlib.pyplot as pltfrom sklearn.decomposition import PCAfrom sklearn.preprocessing import StandardScalerscaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)pca_all = PCA()pca_all.fit(X_train_scaled)individual = pca_all.explained_variance_ratio_cumulative = np.cumsum(individual)plt.figure(figsize=(9, 5))plt.plot(    range(1, len(individual) + 1),    individual,    marker="o",    label="Individual variance",)plt.plot(    range(1, len(cumulative) + 1),    cumulative,    marker="s",    label="Cumulative variance",)plt.axhline(y=0.95, linestyle="--", label="95% variance")plt.xlabel("Number of principal components")plt.ylabel("Explained variance ratio")plt.title("PCA scree plot")plt.xticks(range(1, len(individual) + 1))plt.legend()plt.grid(True)plt.tight_layout()plt.show()
```

Use this plot to inspect the diminishing returns from adding components. Don't choose the number of components based on the plot alone if the ultimate goal is prediction; also compare downstream model performance.

## 8. PCA for visualization: reduce 20 features to 2D

One popular use of PCA is to visualize high-dimensional observations in two dimensions.

```
import matplotlib.pyplot as pltfrom sklearn.decomposition import PCAfrom sklearn.preprocessing import StandardScalerscaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)pca_2d = PCA(n_components=2)X_train_2d = pca_2d.fit_transform(X_train_scaled)plt.figure(figsize=(8, 6))for class_label in np.unique(y_train):    mask = y_train == class_label    plt.scatter(        X_train_2d[mask, 0],        X_train_2d[mask, 1],        label=f"Class {class_label}",        alpha=0.6,    )plt.xlabel("Principal Component 1")plt.ylabel("Principal Component 2")plt.title("Training data projected onto two PCA components")plt.legend()plt.tight_layout()plt.show()
```

This produces a scatter plot in which each observation is represented by PC1 and PC2.

If the classes form separate clusters, the projection may reveal useful structure. If they overlap, it does not necessarily mean the original features cannot distinguish the classes: separation may exist in directions not captured by the first two components.

Important: PCA is unsupervised. It doesn't use `y_train` to find the component directions. We use the labels only to color the plot.

## 9. Compare a model with and without PCA

Reducing dimensionality is useful only if it helps meet the objective. Let's compare logistic regression trained on the original standardized features with logistic regression trained on the PCA representation.

```
from sklearn.linear_model import LogisticRegressionfrom sklearn.pipeline import Pipelinefrom sklearn.decomposition import PCAfrom sklearn.preprocessing import StandardScalerfrom sklearn.metrics import (    accuracy_score,    f1_score,    roc_auc_score,    average_precision_score,)models = {    "Original features": Pipeline([        ("scaler", StandardScaler()),        ("classifier", LogisticRegression(max_iter=2000)),    ]),    "PCA: 5 components": Pipeline([        ("scaler", StandardScaler()),        ("pca", PCA(n_components=5)),        ("classifier", LogisticRegression(max_iter=2000)),    ]),}results = []for name, model in models.items():    model.fit(X_train, y_train)    predictions = model.predict(X_test)    probabilities = model.predict_proba(X_test)[:, 1]    results.append({        "model": name,        "accuracy": accuracy_score(y_test, predictions),        "f1": f1_score(y_test, predictions),        "roc_auc": roc_auc_score(y_test, probabilities),
```

The pipeline ensures that the scaler and PCA are fitted on training data only.

### How should you interpret the results?

- If PCA improves performance, reduces latency, or significantly reduces storage, it may be worthwhile.
- If performance decreases and there is no operational benefit, keep the original features.
- If performance is similar but the reduced representation is faster or cheaper, PCA may still be useful.
- If interpretability matters, PCA may be a poor trade-off because the components mix multiple original features.

This example evaluates on the test set for illustration. In a real model-development process, use cross-validation or a validation set to choose the component count and model. Reserve the test set for the final evaluation after those decisions are made.

## 10. Reconstruct the original features using `inverse_transform()`

PCA can approximately reconstruct the original data from its reduced representation.

```
from sklearn.decomposition import PCAfrom sklearn.preprocessing import StandardScalerfrom sklearn.metrics import mean_squared_errorscaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)X_test_scaled = scaler.transform(X_test)pca = PCA(n_components=5)X_train_reduced = pca.fit_transform(X_train_scaled)X_test_reduced = pca.transform(X_test_scaled)# Reconstruct the standardized test dataX_test_reconstructed = pca.inverse_transform(X_test_reduced)reconstruction_error = mean_squared_error(    X_test_scaled,    X_test_reconstructed,)print("Reconstruction MSE:", reconstruction_error)
```

When fewer components are retained, some information is lost, so reconstruction is generally approximate. Retaining all components (for full-rank data) can reconstruct the input up to numerical precision.

Notice that reconstruction here is performed in the standardized feature space. If you need approximate values in the original units, apply:

```
X_test_reconstructed_original = scaler.inverse_transform(    X_test_reconstructed)
```

This is useful for understanding compression loss, but it does not make PCA a feature-selection technique. The reconstructed columns are estimates derived from the compressed components.

## 11. Production-ready PCA pipeline

For production, you should avoid manually applying the scaler, PCA, and model in separate places. Put them into a single pipeline so the exact same transformation is used during training and inference.

```
from sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.decomposition import PCAfrom sklearn.linear_model import LogisticRegressionproduction_pipeline = Pipeline([    ("scaler", StandardScaler()),    ("pca", PCA(n_components=0.95)),    (        "classifier",        LogisticRegression(            max_iter=2000,            random_state=42,        ),    ),])# Trainingproduction_pipeline.fit(X_train, y_train)# Inference on new observations with the same input columnspredictions = production_pipeline.predict(X_test)probabilities = production_pipeline.predict_proba(X_test)[:, 1]print("Predictions:", predictions[:10])print("Probabilities:", probabilities[:10])print(    "PCA components:",    production_pipeline.named_steps["pca"].n_components_,)
```

The pipeline learns the scaling parameters, PCA components, and classifier during `fit()`. During inference, it applies the stored transformations in order.

For a real production system, you would additionally:

- Version the model and feature schema.
- Validate input columns, types, missing values, and feature ordering.
- Monitor feature distributions and prediction quality.
- Test the serialized pipeline after loading it in the deployment environment.
- Use time-aware or group-aware validation if the observations are financial transactions or repeated customer records.

PCA can be saved with a fitted scikit-learn pipeline using tools such as `joblib`, subject to normal model-artifact security and versioning practices.

## 12. PCA limitations you should know

| Limitation               | Why it matters                                                          |
| ------------------------ | ----------------------------------------------------------------------- |
| Linear transformation    | PCA may not capture nonlinear structure effectively                     |
| Unsupervised objective   | High variance doesn't necessarily mean high predictive value            |
| Reduced interpretability | Components mix original features                                        |
| Scale sensitivity        | Numeric scaling often matters substantially                             |
| Outlier sensitivity      | Extreme observations can strongly influence variance                    |
| Information loss         | Fewer components mean approximate reconstruction                        |
| Categorical data         | Standard PCA is not directly suitable for arbitrary categorical columns |

For nonlinear structure, you might explore kernel PCA or other dimensionality-reduction methods. For visualization, t-SNE and UMAP are also common, though their embeddings should not be interpreted as preserving every global distance or as automatically improving predictive models.

For categorical or mixed tabular data, select an appropriate preprocessing strategy before dimensionality reduction rather than blindly passing encoded columns into PCA.

## 13. PCA vs. other dimensionality-reduction techniques

| Technique     | Best-known use                                          | Main consideration                                 |
| ------------- | ------------------------------------------------------- | -------------------------------------------------- |
| PCA           | Linear compression, preprocessing, visualization        | Components maximize variance                       |
| Truncated SVD | Sparse, high-dimensional matrices such as text features | Can operate without centering sparse matrices      |
| Kernel PCA    | Capturing some nonlinear structure                      | More computationally demanding                     |
| t-SNE         | Visualizing local neighborhood structure                | Mainly exploratory visualization                   |
| UMAP          | Visualization and some embedding workflows              | Results depend on configuration and data structure |
| Autoencoder   | Learning nonlinear compressed representations           | Requires training and careful validation           |

For text classification or semantic retrieval, PCA is not automatically the right choice: methods such as Truncated SVD can be more suitable for sparse TF-IDF features, while embedding-based systems may use their original vectors or a validated compression method.

## 14. Senior ML engineer interview answer

If an interviewer asks, “What is PCA and when would you use it?”, you can answer:

> PCA is an unsupervised linear dimensionality-reduction technique that transforms correlated numeric features into orthogonal principal components ordered by explained variance. I typically standardize the numeric training features, fit PCA on the training set, and apply the learned transformation to validation, test, and inference data.
>
> I choose the number of components using explained variance as an initial heuristic, then validate the downstream model's performance and operational benefits. I compare against a baseline without PCA because retaining most of the variance does not guarantee retaining the most predictive information.
>
> In production, I keep scaling, PCA, and the model in a single versioned pipeline to prevent training-serving inconsistencies. I also consider interpretability, outliers, data drift, and whether the model actually benefits from dimensionality reduction.

Remember the distinction: feature selection chooses existing columns; PCA creates new components from combinations of existing columns. PCA is most valuable when that compressed representation provides a measurable benefit for the task you're solving.