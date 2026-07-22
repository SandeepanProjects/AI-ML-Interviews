# Temperature vs Top-p (Nucleus Sampling) — Senior AI Engineer Explanation

Both **Temperature** and **Top-p** control **how GPT chooses the next token** during text generation.

They **do not change what the model knows**.

They only change **how the next token is selected** from the probability distribution.

---

# Where They Fit in the Pipeline

```text
Input Prompt
      │
      ▼
Transformer
      │
      ▼
Logits
      │
      ▼
Temperature
      │
      ▼
Softmax
      │
      ▼
Top-p Filtering
      │
      ▼
Sample Next Token
```

Notice:

* **Temperature** modifies the probability distribution.
* **Top-p** filters which tokens are allowed to be sampled.

---

# Example

Suppose GPT predicts these logits:

| Token  | Logit |
| ------ | ----: |
| Paris  |  10.0 |
| London |   8.0 |
| Berlin |   6.0 |
| Rome   |   4.0 |
| Mars   |   1.0 |

These logits are not probabilities.

---

# Step 1: Convert to Probabilities

```python
import torch

logits = torch.tensor([10.0, 8.0, 6.0, 4.0, 1.0])

probs = torch.softmax(logits, dim=0)

print(probs)
```

Output (approximately):

```text
Paris   0.864
London  0.117
Berlin  0.016
Rome    0.002
Mars    0.0001
```

Normally GPT would almost always choose **Paris**.

---

# What is Temperature?

Temperature changes the **shape of the probability distribution**.

Formula:

[
P_i=\text{Softmax}\left(\frac{\text{logits}}{T}\right)
]

where:

* (T < 1): Sharper distribution
* (T = 1): Original distribution
* (T > 1): Flatter distribution

---

# Low Temperature (T = 0.2)

```python
import torch

logits = torch.tensor([10.0, 8.0, 6.0])

temperature = 0.2

probs = torch.softmax(logits / temperature, dim=0)

print(probs)
```

Approximate output:

```text
Paris   0.99995
London  0.00005
Berlin  ~0
```

Almost deterministic.

---

## Visualization

```text
Paris      ███████████████████

London     ▏

Berlin
```

GPT almost always generates **Paris**.

---

# High Temperature (T = 2.0)

```python
temperature = 2.0

probs = torch.softmax(logits / temperature, dim=0)

print(probs)
```

Approximate output:

```text
Paris   0.665
London  0.245
Berlin  0.090
```

Now lower-probability tokens have a much better chance of being sampled.

---

## Visualization

```text
Paris      ███████████

London     █████

Berlin     ██
```

Generation becomes more diverse.

---

# Temperature Summary

| Temperature         | Behavior                           |
| ------------------- | ---------------------------------- |
| 0.0 (or very close) | Nearly deterministic (greedy-like) |
| 0.2                 | Highly focused                     |
| 0.7                 | Balanced (common default)          |
| 1.0                 | Original distribution              |
| 1.5                 | Creative                           |
| 2.0+                | Very random; quality may decrease  |

---

# What is Top-p (Nucleus Sampling)?

Top-p does **not** change probabilities.

Instead, it selects the **smallest set of tokens whose cumulative probability exceeds p**.

Suppose:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.50 |
| London |        0.25 |
| Berlin |        0.15 |
| Rome   |        0.07 |
| Mars   |        0.03 |

---

## Top-p = 0.80

Cumulative probabilities:

| Token  | Cumulative |
| ------ | ---------: |
| Paris  |       0.50 |
| London |       0.75 |
| Berlin |       0.90 |

The smallest set whose cumulative probability exceeds **0.80** is:

```text
Paris
London
Berlin
```

Only these tokens remain.

The others are discarded.

---

# Code Example

```python
import torch

tokens = ["Paris", "London", "Berlin", "Rome", "Mars"]

probs = torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03])

sorted_probs, indices = torch.sort(probs, descending=True)

cum = torch.cumsum(sorted_probs, dim=0)

print(sorted_probs)
print(cum)
```

Output:

```text
0.50
0.25
0.15
0.07
0.03

0.50
0.75
0.90
0.97
1.00
```

For `top_p = 0.8`, only the first three tokens are eligible for sampling.

---

# Visualization

Original distribution:

```text
Paris      ██████████

London     █████

Berlin     ███

Rome       █

Mars       ▏
```

After `top_p = 0.8`:

```text
Paris      ██████████

London     █████

Berlin     ███

-----------------

Rome       ❌

Mars       ❌
```

---

# Temperature vs Top-p

### Temperature

Changes probabilities.

```text
Before

Paris   0.70

London  0.20

Berlin  0.10
```

After higher temperature:

```text
Paris   0.50

London  0.30

Berlin  0.20
```

Everyone gets a better chance.

---

### Top-p

Keeps probabilities the same, but removes unlikely candidates.

```text
Before

Paris

London

Berlin

Rome

Mars
```

After `top_p = 0.8`:

```text
Paris

London

Berlin

Rome ❌

Mars ❌
```

---

# Using Both Together

Most production systems combine them.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer(
    "Artificial Intelligence",
    return_tensors="pt"
)

output = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Typical settings:

```python
temperature = 0.7
top_p = 0.9
```

These often provide a good balance between coherence and diversity.

---

# Production Settings

## Chatbot

```python
temperature = 0.2
top_p = 0.9
```

Characteristics:

* Stable
* Accurate
* Predictable

---

## Creative Writing

```python
temperature = 1.2
top_p = 0.95
```

Characteristics:

* Diverse
* Imaginative
* More risk of inconsistencies

---

## Code Generation

```python
temperature = 0.1
top_p = 0.95
```

Characteristics:

* Deterministic
* Reproducible
* Lower chance of syntactic errors

---

# Common Misconception

Many people think:

```text
Higher temperature means GPT becomes smarter.
```

This is incorrect.

Higher temperature only increases randomness. It does **not** improve the model's reasoning ability or factual knowledge.

---

# Interview Questions

1. What is the difference between logits and probabilities?
2. How does temperature affect the softmax distribution?
3. What happens when temperature approaches zero?
4. What is nucleus (top-p) sampling?
5. How is top-p different from top-k sampling?
6. Why are temperature and top-p often used together?
7. Which decoding settings would you choose for a customer support chatbot? Why?
8. Why can a high temperature increase hallucinations?
9. Can top-p improve factual accuracy? Why or why not?
10. How would you tune decoding parameters differently for code generation, summarization, and creative writing?
