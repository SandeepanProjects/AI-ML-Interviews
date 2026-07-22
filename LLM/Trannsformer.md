# Transformer Architecture End-to-End (Senior AI Engineer Level)

The Transformer is the foundation of almost every modern LLM (GPT, Llama, Claude, Gemini, Mistral, DeepSeek, etc.).

Unlike RNNs or LSTMs, Transformers process all tokens **in parallel** using **Self-Attention** instead of sequential computation.

---

# Overall Pipeline

```
                    INPUT TEXT
                         │
                         ▼
                Tokenization (BPE)
                         │
                         ▼
                 Token IDs (integers)
                         │
                         ▼
                 Embedding Layer
                         │
                         ▼
             Positional Encoding Added
                         │
                         ▼
     ┌─────────────────────────────────────┐
     │         Transformer Blocks          │
     │                                     │
     │  Multi Head Attention               │
     │       +                             │
     │  Add & LayerNorm                    │
     │       +                             │
     │  Feed Forward Network               │
     │       +                             │
     │  Add & LayerNorm                    │
     └─────────────────────────────────────┘
                         │
                  repeated N times
                         │
                         ▼
                  Final Hidden State
                         │
                         ▼
                Linear Projection
                         │
                         ▼
                     Softmax
                         │
                         ▼
                Probability Distribution
                         │
                         ▼
                 Next Token Prediction
```

---

# Example

Sentence

```
I love artificial intelligence
```

After tokenization

```
["I",
 "love",
 "artificial",
 "intelligence"]
```

Vocabulary IDs

```
[12, 54, 786, 4501]
```

---

# Step 1 — Tokenization

Transformers cannot understand text.

They only understand integers.

Example

```
Text

Hello World

↓

Tokenizer

["Hello","World"]

↓

IDs

[15496, 2159]
```

GPT uses

```
Byte Pair Encoding (BPE)
```

Llama uses

```
SentencePiece
```

---

# Step 2 — Embedding Layer

Each token becomes a dense vector.

Suppose

Vocabulary

```
50000 words
```

Embedding dimension

```
768
```

Embedding matrix

```
50000 × 768
```

Each word

```
15496

↓

[0.12,
-0.44,
0.88,
...
768 values]
```

Shape

```
Input

(batch, seq)

↓

(batch, seq, embedding_dim)

Example

(32,128)

↓

(32,128,768)
```

PyTorch

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(50000, 768)

tokens = torch.randint(0,50000,(2,5))

x = embedding(tokens)

print(x.shape)
```

Output

```
torch.Size([2,5,768])
```

---

# Problem

Embeddings know

```
what
```

They don't know

```
where
```

Example

```
Dog bites man

Man bites dog
```

Same words

Different meaning.

Need position information.

---

# Step 3 — Positional Encoding

We add positional vectors.

```
Embedding

+

Position Encoding

=

Final Input
```

Example

```
Token

Dog

Embedding

[0.2 0.4 ...]

Position vector

[0.8 0.1 ...]

↓

New embedding

[1.0 0.5 ...]
```

Original Transformer uses sinusoidal encodings:

[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
]

[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
]

Modern LLMs often use **RoPE (Rotary Positional Embeddings)** instead.

---

# Step 4 — Self Attention

This is the heart of the Transformer.

Suppose

```
The animal didn't cross the road because it was tired.
```

Question

```
What does "it" refer to?
```

Self-attention allows "it" to attend to "animal".

Instead of only the previous word, every token can look at every other token.

---

## Query, Key, Value

Each embedding is projected into three vectors:

```
Embedding
    │
    ├──► Query (Q)
    ├──► Key   (K)
    └──► Value (V)
```

Using learned linear layers:

[
Q = XW_Q,\quad
K = XW_K,\quad
V = XW_V
]

Where:

* (X): input embeddings
* (W_Q, W_K, W_V): trainable matrices

---

## Attention Score

Similarity between tokens:

[
Score = QK^T
]

Higher score ⇒ stronger attention.

---

## Scaling

Large dot products can destabilize softmax, so scale by:

[
\frac{QK^T}{\sqrt{d_k}}
]

where (d_k) is the key dimension.

---

## Softmax

Convert scores to probabilities:

[
Attention = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
]

---

## Weighted Sum

The output is:

[
\text{Output} = \text{Attention} \times V
]

Overall attention equation:

[
\boxed{
\text{Attention}(Q,K,V)=
\text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
]

---

## PyTorch Implementation

```python
import math
import torch

Q = torch.randn(4,64)
K = torch.randn(4,64)
V = torch.randn(4,64)

scores = Q @ K.T

scores = scores / math.sqrt(64)

weights = torch.softmax(scores, dim=-1)

output = weights @ V
```

---

# Step 5 — Multi-Head Attention

Instead of one attention mechanism, use multiple heads.

```
Head1

Grammar

Head2

Pronouns

Head3

Verbs

Head4

Objects
```

Each head learns different relationships.

```
Input
      │
 ┌────┴─────┐
Head1 Head2 Head3 Head4
 └────┬─────┘
      ▼
 Concatenate
      ▼
 Linear Layer
```

Benefits:

* Different semantic relationships
* Long-range dependencies
* Richer representations

---

# Step 6 — Add & LayerNorm (Residual Connection)

Transformer uses skip connections.

```
Input
  │
Attention
  │
 + Input
  │
LayerNorm
```

Mathematically:

[
Y=\text{LayerNorm}(X+\text{Attention}(X))
]

Benefits:

* Better gradient flow
* Stable training
* Faster convergence

---

# Step 7 — Feed Forward Network (FFN)

Each token is processed independently by an MLP.

Typical structure:

```
768

↓

3072

↓

768
```

Formula:

[
FFN(x)=W_2(\text{GELU}(W_1x+b_1))+b_2
]

PyTorch:

```python
ffn = nn.Sequential(
    nn.Linear(768,3072),
    nn.GELU(),
    nn.Linear(3072,768)
)
```

---

# Step 8 — Add & LayerNorm Again

Another residual connection:

[
Y=\text{LayerNorm}(X+FFN(X))
]

---

# Step 9 — Stack Transformer Blocks

A Transformer block contains:

```
Input
  │
Multi-Head Attention
  │
Residual
  │
LayerNorm
  │
Feed Forward
  │
Residual
  │
LayerNorm
```

Repeat this many times.

Examples:

| Model       | Layers |
| ----------- | ------ |
| GPT-2 Small | 12     |
| GPT-2 XL    | 48     |
| Llama 2 7B  | 32     |
| Llama 3 8B  | 32     |
| GPT-3 175B  | 96     |

---

# Step 10 — Output Projection

The final hidden state is projected back to the vocabulary size.

```
Hidden

768

↓

Linear

768 → 50000
```

Produces one logit per vocabulary token.

---

# Step 11 — Softmax

Convert logits to probabilities:

```python
probs = torch.softmax(logits, dim=-1)
```

Example:

```
dog      0.72

cat      0.14

house    0.03

apple    0.01
```

The model predicts the most likely next token (or samples from this distribution, depending on decoding strategy).

---

# Encoder vs Decoder

| Feature                                    | Encoder | Decoder                 |
| ------------------------------------------ | ------- | ----------------------- |
| Bidirectional attention                    | ✅       | ❌ (uses causal masking) |
| Predict next token                         | ❌       | ✅                       |
| Used in BERT                               | ✅       | ❌                       |
| Used in GPT                                | ❌       | ✅                       |
| Used in Translation (original Transformer) | ✅       | ✅                       |

* **Encoder-only** (e.g., BERT): best for understanding tasks like classification and question answering.
* **Decoder-only** (e.g., GPT, Llama): best for text generation.
* **Encoder–decoder** (e.g., T5, original Transformer): best for sequence-to-sequence tasks such as translation and summarization.

---

# Training Pipeline

```
Text Corpus
      │
      ▼
Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Embedding + Position
      │
      ▼
Transformer Layers
      │
      ▼
Logits
      │
      ▼
Softmax
      │
      ▼
Cross Entropy Loss
      │
      ▼
Backpropagation
      │
      ▼
AdamW Optimizer
      │
      ▼
Updated Weights
```

---

# Computational Complexity

For a sequence length (n) and hidden size (d):

| Component      | Time Complexity |
| -------------- | --------------- |
| Self-Attention | (O(n^2 d))      |
| Feed Forward   | (O(nd^2))       |

The quadratic (O(n^2)) attention cost is the main bottleneck for long contexts, motivating variants such as FlashAttention, sparse attention, linear attention, and grouped-query attention.

---

# Why Transformers Replaced RNNs/LSTMs

| Feature                          | RNN/LSTM  | Transformer |
| -------------------------------- | --------- | ----------- |
| Parallel processing              | ❌         | ✅           |
| Long-range dependencies          | Limited   | Excellent   |
| GPU utilization                  | Poor      | Excellent   |
| Training speed                   | Slow      | Fast        |
| Scales to billions of parameters | Difficult | Excellent   |
| Foundation of modern LLMs        | ❌         | ✅           |

---

# Common Interview Questions

1. Why are Query, Key, and Value matrices needed?
2. Why divide attention scores by (\sqrt{d_k})?
3. Why use multiple attention heads instead of one?
4. What is the role of positional encoding?
5. Why are residual connections and layer normalization important?
6. What is causal masking, and why is it required in GPT-style models?
7. Why does self-attention have quadratic complexity?
8. What is the difference between encoder-only, decoder-only, and encoder–decoder Transformers?
9. How does multi-head attention differ from single-head attention?
10. How do modern optimizations such as FlashAttention and RoPE improve Transformer performance?

Mastering these concepts gives you a solid foundation for understanding how modern LLMs are built, trained, and optimized in production systems.
