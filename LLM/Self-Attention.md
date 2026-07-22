# Self-Attention Explained (Senior AI Engineer Level)

Self-attention is **the core algorithm behind Transformers**. It allows every token in a sequence to decide **which other tokens are most relevant** when computing its representation.

Instead of processing words one-by-one like an RNN, self-attention lets each token "look at" all the other tokens simultaneously.

---

# Example

Sentence:

```text
The animal didn't cross the road because it was tired.
```

Question:

```text
What does "it" refer to?
```

Humans know:

```text
it → animal
```

A Transformer learns this relationship using self-attention.

---

# High-Level Idea

Suppose we have:

```text
I love AI
```

Each word asks:

* Which words should I pay attention to?
* How much attention should I give each one?
* Combine information from those words to create a better representation.

Example:

```text
          I   love   AI

I       0.7   0.2   0.1

love    0.3   0.5   0.2

AI      0.1   0.4   0.5
```

Every row sums to **1**.

These are the attention weights.

---

# Step 1 — Input Embeddings

Suppose:

```text
Sentence

I love AI
```

After embedding:

```text
I

[1.0, 0.5]

love

[0.3, 0.8]

AI

[0.9, 0.2]
```

Matrix:

[
X=
\begin{bmatrix}
1.0 & 0.5\
0.3 & 0.8\
0.9 & 0.2
\end{bmatrix}
]

Shape:

```text
(number_of_tokens, embedding_dimension)

(3,2)
```

---

# Step 2 — Create Query, Key, and Value

Every embedding is projected into **three different vectors**.

```
Embedding
     │
 ┌───┼─────────┐
 │   │         │
 ▼   ▼         ▼
Query Key     Value
```

Why three?

* **Query (Q)**: What information am I looking for?
* **Key (K)**: What information do I contain?
* **Value (V)**: What information should I contribute?

Think of a search engine:

```
Google Search

Search text

↓

Query

Documents

↓

Keys

Document content

↓

Values
```

---

# Code

```python
import torch

X = torch.tensor([
    [1.0, 0.5],
    [0.3, 0.8],
    [0.9, 0.2]
])
```

Weight matrices:

```python
Wq = torch.randn(2,2)
Wk = torch.randn(2,2)
Wv = torch.randn(2,2)
```

Generate Q, K, V:

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv

print(Q)
print(K)
print(V)
```

Shapes:

```text
Q = (3,2)

K = (3,2)

V = (3,2)
```

---

# Step 3 — Compute Similarity Scores

Now compare every query with every key.

Formula:

[
Scores = QK^T
]

Code:

```python
scores = Q @ K.T

print(scores)
```

Suppose:

```text
scores

[[2.1 1.5 0.3]
 [1.8 3.4 1.1]
 [0.9 1.2 2.8]]
```

Interpretation:

```
Token1 compared with

Token1

Token2

Token3
```

Large value

↓

High similarity

---

# Why Dot Product?

Imagine vectors:

```text
Token A

→

Token B

→

```

Similar direction

↓

Large dot product

Different direction

↓

Small dot product

The dot product measures how well the query and key align.

---

# Step 4 — Scale the Scores

Transformer divides by

[
\sqrt{d_k}
]

Formula:

[
\frac{QK^T}{\sqrt{d_k}}
]

Why?

If vectors are high-dimensional, dot products become large. Feeding very large values into softmax makes it nearly one-hot, causing tiny gradients and unstable learning.

Code:

```python
import math

dk = K.shape[-1]

scores = scores / math.sqrt(dk)
```

---

# Step 5 — Apply Softmax

Convert scores into probabilities.

Code:

```python
weights = torch.softmax(scores, dim=-1)

print(weights)
```

Example:

```text
[[0.60 0.30 0.10]

 [0.20 0.50 0.30]

 [0.10 0.20 0.70]]
```

Properties:

Every row

```text
sums to 1
```

These are attention weights.

---

# Visualization

```
Token

I

↓

looks at

I

60%

love

30%

AI

10%
```

Another token:

```
AI

↓

looks at

I

10%

love

20%

AI

70%
```

Each token decides how much information to gather from every other token.

---

# Step 6 — Multiply by Value Matrix

Formula:

[
Output = Attention \times V
]

Code:

```python
output = weights @ V

print(output)
```

Shape:

```text
(3,2)
```

Each output row is a weighted combination of all value vectors.

---

# Complete Mathematical Formula

[
\boxed{
Attention(Q,K,V)
================

Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
}
]

This is the core computation of self-attention.

---

# Full PyTorch Implementation

```python
import math
import torch

# Input embeddings
X = torch.tensor([
    [1.0, 0.5],
    [0.3, 0.8],
    [0.9, 0.2]
])

# Trainable weights
Wq = torch.randn(2, 2)
Wk = torch.randn(2, 2)
Wv = torch.randn(2, 2)

# Generate Query, Key, Value
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Attention scores
scores = Q @ K.T

# Scale
scores = scores / math.sqrt(K.shape[-1])

# Softmax
weights = torch.softmax(scores, dim=-1)

# Final output
output = weights @ V

print("Q")
print(Q)

print("\nK")
print(K)

print("\nV")
print(V)

print("\nAttention Scores")
print(scores)

print("\nAttention Weights")
print(weights)

print("\nOutput")
print(output)
```

---

# What Happens Internally?

```
Input Embeddings
       │
       ▼
Linear Layer
       │
 ┌─────┼─────┐
 │     │     │
 ▼     ▼     ▼
Q      K      V
 │
 │
 ▼
Q × Kᵀ
 │
 ▼
Scale by √dₖ
 │
 ▼
Softmax
 │
 ▼
Attention Weights
 │
 ▼
Weights × V
 │
 ▼
Output
```

---

# Example: Why Self-Attention Works

Sentence:

```text
The cat sat on the mat because it was tired.
```

For the token **"it"**, the attention weights might look like:

| Token   | Weight |
| ------- | ------ |
| The     | 0.02   |
| cat     | 0.58   |
| sat     | 0.04   |
| on      | 0.01   |
| the     | 0.01   |
| mat     | 0.06   |
| because | 0.08   |
| it      | 0.15   |
| was     | 0.03   |
| tired   | 0.02   |

The highest weight is on **"cat"**, helping the model infer that **"it"** refers to the cat.

---

# Complexity

For a sequence length (n):

* Computing (QK^T) compares every token with every other token.
* Time complexity: (O(n^2 d))
* Memory complexity: (O(n^2))

This quadratic scaling is why very long contexts become expensive and why techniques like **FlashAttention**, **sliding-window attention**, and **linear attention** have been developed.

---

# Common Interview Questions

1. Why are **Query**, **Key**, and **Value** separate projections instead of using the embeddings directly?
2. Why is the attention score divided by (\sqrt{d_k})?
3. Why does each row of the attention matrix sum to 1?
4. Why multiply the attention weights by the **Value** matrix instead of the **Key** matrix?
5. What are the shapes of (Q), (K), (V), the score matrix, and the output for an input of shape `(batch_size, seq_len, d_model)`?
6. Why is self-attention more parallelizable than RNNs?
7. Why does self-attention have quadratic complexity with respect to sequence length?
8. How do **causal masking** (used in GPT) and **padding masks** modify the attention computation?
