# Implement Gradient Descent from Scratch

Gradient Descent is the **most important optimization algorithm in Machine Learning and Deep Learning**.

It is used to train:

* Linear Regression
* Logistic Regression
* Neural Networks
* CNNs
* Transformers
* LLMs (GPT, Llama, Claude)
* Diffusion Models

Almost every deep learning model is trained using some variant of gradient descent.

---

# What Problem Does Gradient Descent Solve?

Suppose we want to fit a line:

[
y = wx + b
]

Our goal is to find the best values of:

* **w** (weight)
* **b** (bias)

that minimize prediction error.

Example dataset:

|  x |  y |
| -: | -: |
|  1 |  2 |
|  2 |  4 |
|  3 |  6 |
|  4 |  8 |

The ideal line is:

```text
y = 2x
```

Initially, we don't know that.

Gradient descent helps us discover it automatically.

---

# High-Level Idea

```text
Random Parameters
       │
       ▼
Predict
       │
       ▼
Compute Error (Loss)
       │
       ▼
Compute Gradient
       │
       ▼
Update Parameters
       │
       ▼
Repeat Until Converged
```

---

# Step 1 — Initialize Parameters

Start with random values.

```python
w = 0.5
b = 0.0
```

---

# Step 2 — Make Predictions

Model:

[
\hat y = wx + b
]

Example:

```python
w = 0.5
b = 0

x = 4

prediction = 0.5 * 4
```

Output

```text
2
```

Actual value

```text
8
```

The prediction is poor.

---

# Step 3 — Compute Loss

We need a way to measure error.

The most common loss for regression is **Mean Squared Error (MSE)**:

[
J(w,b)=\frac1n\sum_{i=1}^{n}(\hat y_i-y_i)^2
]

Example:

Predictions

```text
2
3
4
5
```

Actual

```text
2
4
6
8
```

Squared errors

```text
0
1
4
9
```

Average

```text
3.5
```

Loss = **3.5**

Lower is better.

---

# Step 4 — Compute Gradients

The gradient tells us:

> Which direction should we move the parameters to reduce the loss?

For linear regression with MSE:

[
\frac{\partial J}{\partial w}
=============================

\frac{2}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)x_i
]

[
\frac{\partial J}{\partial b}
=============================

\frac{2}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)
]

A positive gradient means the parameter is too large.

A negative gradient means the parameter is too small.

---

# Step 5 — Update Parameters

Update rule:

[
w = w - \alpha \frac{\partial J}{\partial w}
]

[
b = b - \alpha \frac{\partial J}{\partial b}
]

Where:

* **α** = learning rate

Example

```python
w = 2.5

gradient = 0.4

learning_rate = 0.1

w = w - learning_rate * gradient
```

Result

```text
2.46
```

Repeat this process until the loss stops decreasing.

---

# Implement Gradient Descent from Scratch

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Initialize parameters
w = 0.0
b = 0.0

learning_rate = 0.1
epochs = 1000

n = len(X)

for epoch in range(epochs):

    # Forward pass
    predictions = w * X + b

    # Compute loss
    loss = np.mean((predictions - y) ** 2)

    # Compute gradients
    dw = (2 / n) * np.sum((predictions - y) * X)
    db = (2 / n) * np.sum(predictions - y)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} "
            f"Loss={loss:.6f} "
            f"w={w:.4f} "
            f"b={b:.4f}"
        )

print("\nFinal Parameters")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
```

Example output

```text
Epoch    0 Loss=30.000000 w=3.0000 b=1.0000
Epoch  100 Loss=0.000002 w=2.0000 b=0.0001
...
Final Parameters
w = 2.0000
b = 0.0000
```

The algorithm learns the correct line:

[
y = 2x
]

---

# Pure Python Implementation

```python
def gradient_descent(X, y, lr=0.01, epochs=1000):

    w = 0.0
    b = 0.0

    n = len(X)

    for epoch in range(epochs):

        dw = 0.0
        db = 0.0
        loss = 0.0

        for xi, yi in zip(X, y):

            prediction = w * xi + b

            error = prediction - yi

            loss += error ** 2

            dw += error * xi
            db += error

        loss /= n

        dw = (2 / n) * dw
        db = (2 / n) * db

        w -= lr * dw
        b -= lr * db

    return w, b


X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

w, b = gradient_descent(X, y)

print(w, b)
```

---

# Batch vs Stochastic vs Mini-Batch Gradient Descent

## 1. Batch Gradient Descent

Uses the **entire dataset** before updating parameters.

```text
Entire Dataset
      │
      ▼
Compute Gradient
      │
      ▼
Update Once
```

Pros:

* Stable updates
* Accurate gradient

Cons:

* Slow for very large datasets

---

## 2. Stochastic Gradient Descent (SGD)

Updates after **every training example**.

```text
Sample 1 → Update
Sample 2 → Update
Sample 3 → Update
...
```

Pros:

* Fast updates
* Escapes shallow local minima

Cons:

* Noisy optimization path

---

## 3. Mini-Batch Gradient Descent

Uses a small batch (e.g., 32, 64, 128 samples).

```text
Batch of 32
      │
      ▼
Gradient
      │
      ▼
Update
```

This is the standard approach for training modern neural networks and LLMs because it balances computational efficiency with stable optimization.

---

# Effect of Learning Rate

Small learning rate:

```text
●──●──●──●──●
```

* Slow convergence
* Stable

Large learning rate:

```text
●────────────●────────────●
```

* May overshoot the optimum
* Can diverge

---

# Time Complexity

Let:

* **N** = number of samples
* **D** = number of features
* **E** = number of epochs

For batch gradient descent:

* One epoch: **O(N × D)**
* Total training: **O(E × N × D)**

---

# Production Optimizers

Plain gradient descent is rarely used in deep learning.

Common optimizers include:

| Optimizer      | Key Idea                                                                  |
| -------------- | ------------------------------------------------------------------------- |
| SGD            | Updates using one sample or mini-batch                                    |
| SGD + Momentum | Accelerates convergence by smoothing updates                              |
| RMSProp        | Adapts learning rates using recent gradient magnitudes                    |
| Adam           | Combines Momentum and RMSProp (most widely used)                          |
| AdamW          | Adam with decoupled weight decay; commonly used for Transformers and LLMs |

---

# Gradient Descent in LLM Training

A simplified training loop for a transformer looks like:

```text
Training Data
      │
      ▼
Tokenizer
      │
      ▼
Transformer Forward Pass
      │
      ▼
Loss (Cross-Entropy)
      │
      ▼
Backpropagation
      │
      ▼
Gradients
      │
      ▼
AdamW Optimizer
      │
      ▼
Updated Weights
      │
      ▼
Next Mini-Batch
```

In practice:

* The forward pass computes predictions.
* Backpropagation computes gradients for millions or billions of parameters.
* An optimizer such as **AdamW** updates those parameters.
* This repeats over many epochs until the model converges.

---

# Senior AI Engineer Interview Questions

### Why do we subtract the gradient?

The gradient points in the direction of **steepest increase** of the loss. To minimize the loss, we move in the **opposite direction**, hence:

[
\theta = \theta - \alpha \nabla J(\theta)
]

---

### Why use mini-batches instead of the full dataset?

Mini-batches:

* Fit efficiently on GPUs
* Require less memory
* Provide frequent parameter updates
* Offer a good trade-off between stability and speed

---

### What happens if the learning rate is too high?

* Loss may oscillate or increase
* Training can become unstable
* The optimizer may never converge

---

### Why is Adam preferred over vanilla gradient descent?

Adam adapts the learning rate for each parameter and incorporates momentum, typically leading to faster and more robust convergence on deep neural networks, especially Transformers and LLMs.
