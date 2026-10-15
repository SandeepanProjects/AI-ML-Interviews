# Supervised vs Unsupervised Learning in Machine Learning

As a machine learning engineer, you should understand these concepts from three perspectives:

1. Conceptual understanding: What is the difference, and when do we use each?
2. Implementation: How do we build models using Python and scikit-learn?
3. Production understanding: How would we use them in real-world projects, such as financial risk analysis, fraud detection, and customer segmentation?

Let's learn them step by step, with complete runnable code.

## 1. What is supervised learning?

Supervised learning is a machine learning approach where the model learns from labeled data.

Labeled data means that we already know the correct answer for each training example.

For example, suppose we want to predict whether a financial transaction is fraudulent.

| Transaction amount | International | Fraud? |
| ------------------ | ------------- | ------ |
| ₹500               | No            | No     |
| ₹50,000            | Yes           | Yes    |
| ₹1,200             | No            | No     |
| ₹80,000            | Yes           | Yes    |
| ₹2,000             | No            | No     |

Here, `Fraud?` is the label, or target variable. The model learns patterns from transaction details and their known outcomes.

After training, we can give it a new transaction and ask it to predict whether the transaction is fraudulent.

## How supervised learning works

Labeled training data

Features (X) + Correct answers (y)

Training algorithm

Learns relationships between inputs and outputs

Trained model

Predicts the output for unseen data

### Types of supervised learning

There are two major types.

[Variational Inference: Bayesian Neural Networks — PyMC  documentation](https://images.openai.com/static-rsc-4/tBCaJu5CrTtUQfz7GKw7FeIfoPfveCqWgqM8z6L3wAdRjIiv5ODO7CN1triWE9aJKqDjmkHRnsSSEL55X4UvEYoGdb31XbvgwEH09zRc3viPC9ZxL2ZdLlY1qW7sFjsN8a6VfmQoYTuowHv7_DyglCGORCjbzC3hC24FBubbPuQ?purpose=inline)

1\. Classification

Predicts a category or class.

Examples:

- Fraud / not fraud
- Spam / not spam
- Loan default / no default
- Customer churn / no churn

[Linear Regression — Mathematical intuition and Python implementation | by Rohan Paris | Medium](https://images.openai.com/static-rsc-4/uv6jnf5OwXI_imbMEDsjJz45D5z2sJaRRTPyFkHLAxaQQg5r3wXEEHvFivja1_aClpGr0VDLCXqT__zIPshc5OGHXEEtu1ypR3q8D1aBOjrJ5o-3WNKyu217cVJTCTVZBbxVVVl8W8TzPawkdjcsKPT5HOVVmpFfOiPH-G3-LLI?purpose=inline)

2\. Regression

Predicts a numerical value.

Examples:

- House price
- Stock return estimate
- Sales revenue
- Customer lifetime value

For example, predicting whether a customer will default is classification, while predicting the amount of financial loss is regression.

## 2. What is unsupervised learning?

Unsupervised learning is a machine learning approach where the model receives data without predefined labels and attempts to discover patterns or structure.

Imagine you have customer transaction data, but you do not know which customers belong to which groups.

| Customer | Monthly spending | Transactions per month |
| -------- | ---------------- | ---------------------- |
| A        | ₹5,000           | 10                     |
| B        | ₹6,000           | 12                     |
| C        | ₹80,000          | 40                     |
| D        | ₹75,000          | 35                     |
| E        | ₹8,000           | 9                      |

There is no column that tells the model "premium customer," "regular customer," or "low-spending customer."

An unsupervised algorithm can identify groups of customers with similar spending patterns.

## How unsupervised learning works

Unlabeled data

Features only — no correct answers provided

Clustering or pattern-discovery algorithm

Finds similarities, structure, or unusual observations

Discovered patterns

For example, customer groups or anomaly scores

### Common types of unsupervised learning

| Technique                | Purpose                                               | Example                      |
| ------------------------ | ----------------------------------------------------- | ---------------------------- |
| Clustering               | Groups similar data points                            | Customer segmentation        |
| Dimensionality reduction | Compresses features while preserving useful structure | PCA for financial data       |
| Anomaly detection        | Identifies unusual observations                       | Unusual transaction patterns |
| Association rule mining  | Finds items that frequently occur together            | Products bought together     |

One important distinction: anomaly detection is often taught alongside unsupervised learning, but real implementations can be unsupervised, semi-supervised, or supervised.

## 3. The most important difference

| Aspect                | Supervised learning                                   | Unsupervised learning                                  |
| --------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| Training data         | Labeled                                               | Unlabeled                                              |
| Target variable (`y`) | Available                                             | Usually absent                                         |
| Main goal             | Predict known outcomes                                | Discover hidden patterns                               |
| Common algorithms     | Linear regression, logistic regression, random forest | K-Means, DBSCAN, PCA                                   |
| Evaluation            | Accuracy, precision, recall, F1, RMSE                 | Silhouette score, cluster analysis, domain validation  |
| Example               | Predict whether a customer will default               | Group customers by financial behavior                  |
| Main challenge        | Getting enough reliable labels                        | Determining whether discovered patterns are meaningful |

Remember this simple rule:

- Supervised learning: Learn from examples that include correct answers.
- Unsupervised learning: Discover patterns without being given the correct answers.

Now let's implement both using Python.

## 4. Project 1 — Supervised learning with Python

We'll build a small fraud-classification model using `scikit-learn`.

Our model will learn from historical transactions labeled as fraudulent or legitimate.

### Step 1: Install the libraries

Run this in your terminal:

```
python -m pip install pandas scikit-learn
```

### Step 2: Complete supervised learning code

Create a file called `supervised_learning.py`.

```
        3, 4, 3, 4, 5, 4,        5, 6, 7, 8, 9, 10,        11, 12, 13, 14, 15, 16,    ],    "is_fraud": [        0, 0, 0, 0, 0, 0,        0, 0, 0, 0, 0, 0,        0, 0, 0, 1, 1, 1,        1, 1, 1, 1, 1, 1,    ],})# 2. Separate features (X) from labels (y).X = data[["amount", "transactions_per_day"]]y = data["is_fraud"]# 3. Split into training and testing data.X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.25,    random_state=42,    stratify=y,)# 4. Build a pipeline.# Scaling is fitted only on training data.model = make_pipeline(    StandardScaler(),    LogisticRegression(random_state=42),)# 5. Train the model on labeled examples.model.fit(X_train, y_train)# 6. Evaluate on examples held out from training.y_pred = model.predict(X_test)print("Accuracy:", accuracy_score(y_test, y_pred))print("\nConfusion matrix:")print(confusion_matrix(y_test, y_pred))print("\nClassification report:")print(classification_report(y_test, y_pred, zero_division=0))# 7. Predict on a new transaction.new_transaction = pd.DataFrame({    "amount": [6500],    "transactions_per_day": [12],})prediction = model.predict(new_transaction)[0]probabilities = model.predict_proba(new_transaction)[0]print("\nPredicted class:", prediction)print("Probability by class:", probabilities)if prediction == 1:    print("Model flags the transaction as potentially fraudulent.")else:    print("Model classifies the transaction as legitimate.")
```

This is a runnable educational example. The small, artificially constructed dataset is not suitable for real fraud detection.

### Step 3: Understand what each part does

A. Prepare the dataset

`X` contains the input features: transaction amount and transaction frequency.

`y` contains the known labels: `0` for legitimate and `1` for fraud.

B. Split the data

`train_test_split()` separates training examples from test examples. The model must not learn from the test set.

`stratify=y` helps preserve the class proportions in each split.

C. Scale the features

`StandardScaler()` standardizes numerical features. This is useful for logistic regression because transaction amounts and transaction counts have different scales.

A pipeline prevents the scaler from accidentally learning statistics from the test set.

D. Train the model

`model.fit(X_train, y_train)` learns a relationship between transaction features and known fraud labels.

E. Evaluate the model

`predict()` produces predicted classes, while `predict_proba()` estimates class probabilities. These probabilities are not automatically calibrated.

The classification report includes precision, recall and F1-score.

### Step 4: Understand the evaluation metrics

In fraud detection, accuracy alone is not enough.

| Metric    | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| Accuracy  | Fraction of all predictions that are correct                    |
| Precision | Of transactions flagged as fraud, how many were actually fraud? |
| Recall    | Of all actual fraud cases, how many did the model detect?       |
| F1-score  | Harmonic mean of precision and recall                           |

For financial fraud detection, recall is important when missing a fraudulent transaction is costly, while precision matters because false alarms can inconvenience legitimate customers.

In production, you would also choose a decision threshold based on business costs, assess calibration, monitor drift and evaluate on a realistic, preferably time-based holdout dataset.

## 5. Project 2 — Unsupervised learning with Python

Now let's build a customer segmentation model using K-Means clustering.

In this case, we will deliberately not provide customer group labels. The algorithm must discover groups from the data.

### Step 1: Complete unsupervised learning code

Create `unsupervised_learning.py`.

```
import pandas as pdfrom sklearn.cluster import KMeansfrom sklearn.preprocessing import StandardScalerfrom sklearn.metrics import silhouette_scorefrom sklearn.pipeline import make_pipeline# 1. Customer behavior data.# There is no target column such as "customer_segment".data = pd.DataFrame({    "monthly_spending": [        5000, 6000, 7000, 8000, 9000,        75000, 80000, 85000, 90000, 95000,        20000, 22000, 25000, 28000, 30000,    ],    "transactions_per_month": [        8, 10, 12, 11, 13,        35, 40, 42, 45, 48,        18, 20, 22, 24, 25,    ],})# 2. Select features. No y labels are required.X = data[["monthly_spending", "transactions_per_month"]]# 3. Scale features before calculating distances.scaler = StandardScaler()X_scaled = scaler.fit_transform(X)# 4. Ask K-Means to discover three clusters.model = KMeans(    n_clusters=3,    random_state=42,    n_init=10,)
```

### Step 2: How does K-Means work?

K-Means tries to divide data into a chosen number of clusters, with each cluster represented by a centroid (the center of its points).

[K-means Clustering: Complete Guide with Algorithm, Implementation & Best Practices - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/Hg3AfMrjqjn8WQJI4eE8OAM4KuGI67Oj4-FItYOAazgIkNGgtmw1fPRvJjO74psEe4d1WsmN-o3AUvlhL5rY1QlohlMR4y9-9vAMzd4yde8dbhmOibyTnX0jYUznWNf9ZMJwiKwWfeGD2pC-KiqIOiU_VXKNRclFt3Q00AEywfE?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/kmeans-clustering-complete-guide)

Illustration of K-Means: points are grouped around cluster centers. The actual code above uses two customer features.

The algorithm works approximately as follows:

1. Choose K. We specify `n_clusters=3`, meaning we want three groups.
2. Initialize centroids. K-Means selects initial cluster centers.
3. Assign customers. Each customer is assigned to its nearest centroid in the scaled feature space.
4. Update centroids. The algorithm recalculates each center based on the assigned customers.
5. Repeat. Assignments and centroids are updated until the algorithm converges or reaches its iteration limit.

### Step 3: Understand the output

The algorithm produces cluster IDs such as `0`, `1`, and `2`.

These IDs are arbitrary. Cluster `2` does not inherently mean "premium customer," and cluster `0` does not mean "low-value customer."

You interpret the clusters by examining the average spending and transaction frequency for each group. For this example, you would expect to discover groups broadly corresponding to lower, medium and higher spending patterns.

The silhouette score helps assess how well-separated the clusters are. It ranges from -1 to 1; a higher value generally indicates better separation, but it does not prove that the clusters are useful to the business.

Important: We chose three clusters in advance. K-Means discovers the membership, not the ideal number of clusters or the business meaning of each group.

## 6. How do we choose the number of clusters?

In K-Means, the parameter `n_clusters` must be selected. Two common techniques are the elbow method and silhouette score.

Here's code to compare different values of K:

```
from sklearn.cluster import KMeansfrom sklearn.metrics import silhouette_scorefrom sklearn.preprocessing import StandardScalerX_scaled = StandardScaler().fit_transform(X)for k in range(2, 7):    model = KMeans(        n_clusters=k,        random_state=42,        n_init=10,    )    labels = model.fit_predict(X_scaled)    print(        f"K={k}, "        f"inertia={model.inertia_:.2f}, "        f"silhouette={silhouette_score(X_scaled, labels):.3f}"    )
```

- Elbow method: Look for a point where reducing the within-cluster sum of squares (`inertia_`) gives diminishing returns.
- Silhouette score: Compare how compact and separated the clusters are.
- Business validation: Check whether the resulting groups are actionable, stable and useful.

Neither method automatically identifies the best business segmentation.

## 7. What about supervised regression?

Classification predicts a category. Regression predicts a numerical value. Both are supervised learning because the training data contains known target values.

For example, predicting monthly revenue:

```
import pandas as pdfrom sklearn.linear_model import LinearRegressionfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import mean_absolute_error, mean_squared_errorfrom sklearn.pipeline import make_pipelinefrom sklearn.preprocessing import StandardScalerdata = pd.DataFrame({    "marketing_spend": [100, 150, 200, 250, 300, 350, 400, 450],    "customers": [20, 25, 30, 35, 40, 48, 55, 60],    "revenue": [2000, 2600, 3100, 3700, 4200, 5100, 5900, 6400],})X = data[["marketing_spend", "customers"]]y = data["revenue"]X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.25, random_state=42)model = make_pipeline(    StandardScaler(),    LinearRegression(),)model.fit(X_train, y_train)predictions = model.predict(X_test)print("Actual:", y_test.tolist())print("Predicted:", predictions.round(2).tolist())print("MAE:", mean_absolute_error(y_test, predictions))print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
```

Here, `revenue` is the target variable. The model learns how marketing spend and customer count relate to revenue.

- MAE is the average absolute prediction error.
- RMSE penalizes larger errors more heavily than MAE.

These metrics are measured in the same units as revenue. This example uses synthetic data for learning, not a reliable financial forecast.

## 8. How these techniques are used in real-world AI projects

Consider a financial services platform.

### Example: Financial intelligence platform

Historical financial data

Transactions, customer behavior, account activity and confirmed outcomes.

Supervised ML

- Predict fraud risk
- Predict loan default
- Estimate customer churn

Needs historical labels for training.

Unsupervised ML

- Segment customers
- Detect unusual patterns
- Discover behavior groups

Can learn from unlabeled features.

Production decision system

Combine model outputs with business rules, access controls, monitoring and human review where appropriate.

A practical distinction for production systems is that unsupervised anomaly detection does not automatically mean fraud detection. Unusual behavior can be legitimate. You need investigation, reliable labels and business validation before making consequential decisions.

## 9. How do we run the code?

Save the examples into their respective Python files, then execute:

```
python supervised_learning.py
python unsupervised_learning.py
```

Run the regression example from its own file as well.

You should see evaluation metrics, predicted classes or values, and discovered customer cluster IDs.

## 10. Senior ML engineer interview questions

Here are the questions you should be ready to answer.

1\. Why is fraud detection usually a supervised learning problem?

2\. Can unsupervised learning be used when labels exist?

3\. Why do we scale data before K-Means?

4\. What is the difference between classification and clustering?

5\. Can a model have high accuracy but still be bad at fraud detection?

6\. How would you deploy these models in production?

## Final takeaway

- Supervised learning learns from labeled examples to predict a known target.
- Unsupervised learning discovers structure in data without predefined target labels.
- Classification and regression are supervised learning tasks.
- Clustering and dimensionality reduction are common unsupervised learning techniques.
- Real production ML systems often combine both approaches.

For your next step, I can generate a visual comparison infographic showing supervised vs. unsupervised learning, with the training-data flow, algorithms, and financial AI examples. Which style would you prefer: clean technical diagram, colorful infographic, or interview revision chart?