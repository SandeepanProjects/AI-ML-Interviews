# How GPT Generates Text (End-to-End)

GPT (**Generative Pre-trained Transformer**) generates text **one token at a time**.

It does **not** generate an entire sentence at once.

Instead, it repeatedly performs this cycle:

```text
Input Tokens
      │
      ▼
Transformer
      │
      ▼
Probability Distribution
      │
      ▼
Choose Next Token
      │
      ▼
Append Token
      │
      ▼
Repeat...
```

---

# Example

Prompt:

```text
The capital of France is
```

GPT generates:

```text
The
capital
of
France
is
Paris
.
```

Notice that it predicts **one token at a time**.

---

# Step 1 — User Enters Prompt

Suppose the prompt is:

```text
I love
```

---

# Step 2 — Tokenization

The tokenizer converts text into integer IDs.

```text
"I"

↓

15496

"love"

↓

1842
```

Result:

```python
tokens = [15496, 1842]
```

---

# Step 3 — Embedding Layer

Each token ID becomes a dense vector.

```python
embedding = nn.Embedding(50000, 768)

x = embedding(torch.tensor(tokens))
```

Shape:

```text
(2,768)
```

Meaning:

* 2 tokens
* 768-dimensional embeddings

---

# Step 4 — Positional Encoding

The model adds position information.

```text
Token Embedding

+

Position Embedding

=

Final Input
```

Without this, the model would not know the difference between:

```text
Dog bites man
```

and

```text
Man bites dog
```

---

# Step 5 — Pass Through Transformer Layers

Each Transformer block contains:

```text
Multi-Head Attention

↓

Residual

↓

LayerNorm

↓

Feed Forward

↓

Residual

↓

LayerNorm
```

This process is repeated many times.

Example:

```text
GPT-2 Small

↓

12 layers

GPT-3

↓

96 layers
```

Each layer refines the token representations.

---

# Step 6 — Compute Logits

Suppose:

Vocabulary size:

```text
50000
```

The final hidden state is projected to:

```text
768

↓

50000
```

This produces one score (logit) per vocabulary token.

Example:

```python
logits = model(input_ids).logits
```

Output shape:

```text
(batch_size,
sequence_length,
vocabulary_size)
```

Example:

```text
(1,5,50000)
```

---

# Step 7 — Convert Logits to Probabilities

Softmax converts logits into probabilities.

Formula:

[
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
]

Code:

```python
import torch

logits = torch.tensor([8.1, 6.3, 3.2])

prob = torch.softmax(logits, dim=0)

print(prob)
```

Example output:

```text
Paris      0.82

London     0.12

Berlin     0.05

Other      0.01
```

These probabilities sum to **1**.

---

# Step 8 — Choose Next Token

The model must decide which token to emit.

Several decoding strategies are available.

### Greedy Decoding

Always choose the highest-probability token.

```python
next_token = torch.argmax(prob)
```

Suppose:

```text
Paris

0.82
```

GPT outputs:

```text
Paris
```

---

### Sampling

Instead of always taking the maximum:

```python
next_token = torch.multinomial(prob, 1)
```

This makes generation more diverse.

---

### Temperature

Temperature controls randomness.

```python
temperature = 0.2
```

Low temperature:

```text
More deterministic
```

High temperature:

```python
temperature = 1.5
```

Produces more creative but potentially less reliable outputs.

---

### Top-k Sampling

Only consider the top **k** tokens.

Example:

```python
top_k = 50
```

Everything outside the top 50 candidates is discarded.

---

### Top-p (Nucleus Sampling)

Instead of a fixed number of tokens, choose the smallest set whose cumulative probability exceeds a threshold.

Example:

```python
top_p = 0.9
```

---

# Step 9 — Append the Token

Prompt:

```text
I love
```

Generated:

```text
AI
```

New sequence:

```text
I love AI
```

Now GPT runs **again**.

---

# Step 10 — Repeat

Second pass:

Input:

```text
I love AI
```

Predict:

```text
because
```

New input:

```text
I love AI because
```

Repeat:

```text
I love AI because it helps...
```

Until a stopping condition is reached.

---

# Auto-Regressive Generation

This process is called **auto-regressive generation**.

```text
Token 1

↓

Predict Token 2

↓

Predict Token 3

↓

Predict Token 4

↓

...
```

Each prediction depends on all previously generated tokens.

---

# Why GPT Cannot Generate Everything at Once

Suppose the sentence is:

```text
The weather is sunny today.
```

The model predicts:

```text
The
```

only after seeing the prompt.

Then:

```text
weather
```

depends on:

```text
The
```

Then:

```text
is
```

depends on:

```text
The weather
```

Every token depends on the previous context.

---

# Causal Masking

GPT is a **decoder-only Transformer**.

During generation:

```text
Token1

↓

Can see

Token1
```

```text
Token2

↓

Can see

Token1
Token2
```

```text
Token3

↓

Can see

Token1
Token2
Token3
```

It **cannot** look into future tokens.

Attention mask:

|    | T1 | T2 | T3 | T4 |
| -- | -: | -: | -: | -: |
| T1 |  ✓ |  ✗ |  ✗ |  ✗ |
| T2 |  ✓ |  ✓ |  ✗ |  ✗ |
| T3 |  ✓ |  ✓ |  ✓ |  ✗ |
| T4 |  ✓ |  ✓ |  ✓ |  ✓ |

This lower-triangular mask prevents "cheating" during training and generation.

---

# Complete Generation Pipeline

```text
User Prompt
      │
      ▼
Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Embedding Layer
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Layers
      │
      ▼
Final Hidden State
      │
      ▼
Linear Projection
      │
      ▼
Logits
      │
      ▼
Softmax
      │
      ▼
Choose Next Token
      │
      ▼
Append Token
      │
      ▼
Repeat Until End
```

---

# Simplified PyTorch Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Artificial Intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=30,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The `generate()` method internally performs the loop:

1. Encode the prompt.
2. Run the Transformer.
3. Compute logits.
4. Apply decoding strategy.
5. Select the next token.
6. Append it to the input.
7. Repeat until `max_new_tokens` or an end-of-sequence token is reached.

---

# What Happens Internally During Generation?

Suppose the prompt is:

```text
I love
```

### Iteration 1

```text
Input

I love

↓

Predict

AI
```

### Iteration 2

```text
Input

I love AI

↓

Predict

because
```

### Iteration 3

```text
Input

I love AI because

↓

Predict

it
```

### Iteration 4

```text
Input

I love AI because it

↓

Predict

helps
```

This continues until the model emits an end-of-sequence token or reaches the configured output limit.

---

# KV Cache: Why Generation Is Fast

A naïve implementation would recompute attention for every previously generated token at each step.

Instead, GPT uses a **Key–Value (KV) cache**.

Without KV cache:

```text
Step 1

Process

A

Step 2

Process

A B

Step 3

Process

A B C

Step 4

Process

A B C D
```

Earlier computations are repeated.

With KV cache:

```text
Step 1

Store K,V for A

↓

Step 2

Reuse A

Compute only B

↓

Step 3

Reuse A,B

Compute only C
```

The cached keys and values from previous tokens are reused, significantly reducing computation during autoregressive decoding.

---

# Senior AI Engineer Interview Questions

1. Why does GPT generate one token at a time?
2. What is the difference between **logits** and **probabilities**?
3. Why is the softmax layer needed?
4. What is the role of temperature during decoding?
5. Compare greedy decoding, beam search, top-k, and top-p sampling.
6. What is causal masking, and why is it required?
7. Why is GPT called an **autoregressive** model?
8. How does the KV cache speed up inference?
9. Why does inference latency generally increase with output length?
10. How would you optimize a production LLM server for high-throughput text generation (e.g., batching, KV-cache management, FlashAttention, speculative decoding, quantization)?
