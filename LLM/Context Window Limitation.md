# Context Window Limitation Explained (Senior AI Engineer Level)

One of the biggest limitations of Large Language Models (LLMs) is the **context window**.

The context window determines **how many tokens the model can process at one time**.

Think of it as the model's **working memory**, not its permanent knowledge.

---

# What is a Context Window?

Suppose an LLM supports a **128K-token** context window.

```
Maximum Input Tokens
+
Maximum Output Tokens
≤
128K tokens
```

Everything the model sees in one request must fit within this limit:

* System prompt
* User prompt
* Conversation history
* Retrieved documents (RAG)
* Tool outputs
* Generated response (depending on API accounting)

---

# Example

Suppose the context window is **10 tokens**.

Input:

```text
I love learning Artificial Intelligence using Python every day
```

Tokenized:

```text
I
love
learning
Artificial
Intelligence
using
Python
every
single
day
```

Exactly 10 tokens.

Now add:

```text
today
```

The total becomes:

```text
11 tokens
```

If the model only supports 10 tokens, it cannot process all 11 together.

---

# Why Does This Limitation Exist?

The main reason is **self-attention**.

Recall the attention equation:

[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Suppose:

```
Sequence Length = n
```

Attention computes:

```
n × n
```

comparisons.

Every token attends to every other token.

---

# Example

Sentence:

```text
I love AI
```

Three tokens:

```
I
love
AI
```

Attention matrix:

|      |  I | love | AI |
| ---- | -: | ---: | -: |
| I    |  ✓ |    ✓ |  ✓ |
| love |  ✓ |    ✓ |  ✓ |
| AI   |  ✓ |    ✓ |  ✓ |

Total comparisons:

```
3 × 3 = 9
```

---

Now suppose:

```
1000 tokens
```

Attention matrix:

```
1000 × 1000

=

1,000,000
```

comparisons.

---

Now suppose:

```
100000 tokens
```

Attention matrix:

```
100000 × 100000

=

10,000,000,000
```

**10 billion attention scores**.

This is why long-context inference is expensive.

---

# Computational Complexity

For sequence length (n):

Time complexity:

[
O(n^2 d)
]

Memory complexity:

[
O(n^2)
]

This quadratic growth is the fundamental bottleneck of standard self-attention.

---

# Visualizing Attention

```
Token1 ─────────────┐
                    │
Token2 ───────────┐ │
                  │ │
Token3 ───────┐   │ │
              │   │ │
Token4 ───────┼───┼─┘
              │   │
Token5 ───────┘   │
                  ▼
        Every token attends
        to every other token
```

As the sequence grows, the number of connections grows quadratically.

---

# PyTorch Example

```python
import torch

seq_len = 5
d_model = 8

Q = torch.randn(seq_len, d_model)
K = torch.randn(seq_len, d_model)

scores = Q @ K.T

print(scores.shape)
```

Output:

```text
torch.Size([5, 5])
```

Now increase:

```python
seq_len = 1000
```

Output:

```text
torch.Size([1000,1000])
```

One million attention scores.

---

# Memory Consumption Example

Suppose:

```
Context = 4096 tokens
```

Attention matrix:

```
4096 × 4096

=
16,777,216
```

entries.

Using `float32`:

```
16,777,216 × 4 bytes

≈ 67 MB
```

This is **just one attention matrix for one head**, before accounting for multiple heads, batches, activations, gradients (during training), etc.

---

# What Happens When the Context is Too Long?

Suppose:

```
Maximum context

4096
```

Input:

```
5000 tokens
```

The model cannot process all tokens together.

Applications usually choose one of these strategies:

* Reject the request
* Truncate old tokens
* Summarize earlier context
* Retrieve only relevant information (RAG)

---

# Example: Conversation

```
User:
Question 1

↓

Assistant:
Answer

↓

User:
Question 2

↓

Assistant:
Answer

↓

...
```

Eventually:

```
Conversation

>

Context Window
```

Older messages are often removed or summarized to stay within the limit.

---

# Code Example: Truncation

```python
MAX_CONTEXT = 8

tokens = [
    "I",
    "love",
    "learning",
    "AI",
    "using",
    "Python",
    "every",
    "day",
    "today",
]

tokens = tokens[-MAX_CONTEXT:]

print(tokens)
```

Output:

```text
[
 'love',
 'learning',
 'AI',
 'using',
 'Python',
 'every',
 'day',
 'today'
]
```

The oldest token (`I`) is dropped.

---

# RAG Solves This Problem

Instead of sending:

```
Entire Company Wiki

↓

300000 tokens
```

Send only the relevant documents.

Pipeline:

```
Question
      │
      ▼
Embedding Model
      │
      ▼
Vector Database
      │
      ▼
Top-5 Documents
      │
      ▼
LLM
```

Only a small, relevant subset is included in the prompt.

---

# Chunking

Large document:

```
500 pages
```

Split into chunks:

```
Chunk 1

Chunk 2

Chunk 3

...

Chunk N
```

Each chunk is embedded separately.

Example code:

```python
def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(
            " ".join(words[i:i+chunk_size])
        )

    return chunks
```

---

# Sliding Window

Instead of processing:

```
10000 tokens
```

Use overlapping windows:

```
Window 1

1–1000

↓

Window 2

900–1900

↓

Window 3

1800–2800
```

Example:

```python
def sliding_window(tokens, window=1000, overlap=100):
    windows = []

    step = window - overlap

    for i in range(0, len(tokens), step):
        windows.append(tokens[i:i+window])

    return windows
```

---

# Summarization

Instead of keeping the full conversation:

```
Conversation

↓

Summary

↓

Continue Chat
```

Example:

```python
conversation = """
User: Explain transformers.
Assistant: ...
User: Explain RAG.
Assistant: ...
"""

summary = """
User previously learned:
- Transformers
- Attention
- RAG
"""
```

The summary replaces hundreds or thousands of earlier tokens.

---

# FlashAttention

Standard attention:

```
Q

↓

Attention Matrix

↓

Softmax

↓

Output
```

Stores the full attention matrix in memory.

FlashAttention computes attention in **tiles (blocks)** and fuses operations so the full matrix does not need to be materialized at once.

Benefits:

* Lower memory usage
* Faster GPU execution
* Better utilization of GPU memory bandwidth

It **does not change the theoretical (O(n^2)) attention computation**, but it greatly improves practical performance.

---

# Long-Context Models

Modern models support larger context windows through a combination of architectural improvements and optimized attention implementations.

Examples include models supporting:

* 32K
* 128K
* 200K+
* 1M+ tokens (for some architectures)

Even with these larger windows, processing more tokens still increases latency and compute cost.

---

# Production Architecture

```
              User Question
                     │
                     ▼
             Retrieve Documents
                     │
                     ▼
              Vector Database
                     │
                     ▼
               Top-K Chunks
                     │
                     ▼
              Prompt Builder
                     │
                     ▼
                 LLM
                     │
                     ▼
                 Response
```

Only the most relevant chunks are included in the context window.

---

# Best Practices

* **Use RAG** instead of sending entire document collections.
* **Chunk large documents** into semantically meaningful pieces.
* **Rerank retrieved chunks** before sending them to the LLM.
* **Summarize long conversations** to preserve important information.
* **Keep prompts concise**—remove redundant instructions.
* **Monitor token usage** to control latency and cost.

---

# Senior AI Engineer Interview Questions

1. What is a context window?
2. Why do Transformers have a context window limitation?
3. Why does self-attention have (O(n^2)) memory and time complexity?
4. What happens if the input exceeds the model's context window?
5. How do chunking and sliding windows help process large documents?
6. How does RAG reduce context window pressure?
7. What is FlashAttention, and what problem does it solve?
8. How do long-context models extend usable context without eliminating the computational cost?
9. What are the trade-offs between using a very large context window and using RAG?
10. How would you design a production chatbot that can answer questions over millions of documents despite a finite context window?
