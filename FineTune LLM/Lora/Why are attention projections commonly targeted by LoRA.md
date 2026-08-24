# 1. Why are attention projections commonly targeted by LoRA?

To understand this, first look at what happens inside a Transformer attention layer.

## Transformer attention

For an input representation (X), the model computes:

[
Q = XW_Q
]

[
K = XW_K
]

[
V = XW_V
]

Then:

[
Attention(Q,K,V)
================

softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Finally:

[
Output = Attention(Q,K,V)W_O
]

The projection matrices are:

```text
Wq → Query projection
Wk → Key projection
Wv → Value projection
Wo → Output projection
```

These are typically large learned matrices.

---

## Why target them?

### Reason 1: They strongly control information flow

Attention determines:

> Which tokens should attend to which other tokens?

For example:

```text
"The bank approved the loan because it had..."

          it
          │
          ▼
Which previous word should "it" attend to?
          │
     ┌────┴────┐
     ▼         ▼
   bank       loan
```

The Q and K projections influence the attention pattern:

```text
Q → what information am I looking for?

K → what information is available?
```

The V projection influences:

```text
V → what information/content should be retrieved?
```

By adapting these matrices, LoRA can substantially change how the model:

* focuses on information
* connects concepts
* uses context
* represents task-specific relationships

---

# Why are `q_proj` and `v_proj` especially common?

Historically, many LoRA setups focused on query and value projections because they can provide useful adaptation capacity with relatively few added parameters.

```text
Input
  │
  ├── q_proj ──► "What should I look for?"     ← LoRA
  │
  ├── k_proj ──► "What information exists?"
  │
  ├── v_proj ──► "What information do I use?"  ← LoRA
  │
  └── o_proj ──► Final attention output
```

Adapting `q_proj` can change **how the model queries/contextualizes information**, while adapting `v_proj` can change **what information representation is carried forward**.

However, don't treat Q+V as a universal rule. In modern practice, the best target modules can vary by model and task.

---

# 2. Attention projections are parameter-heavy

Suppose:

```text
hidden_size = 4096
```

Each projection might approximately be:

[
4096 \times 4096
]

One projection:

[
16,777,216
]

parameters.

Four projections:

```text
Q → 16.7M
K → 16.7M
V → 16.7M
O → 16.7M
```

Total:

```text
~67 million parameters per layer
```

With LoRA rank (r=8), each 4096 × 4096 projection gets:

[
r(d_{in}+d_{out})
]

[
8(4096+4096)
============

65,536
]

So you can adapt an influential component with far fewer trainable parameters.

---

# 3. Code: applying LoRA to attention projections

For a Llama-style model:

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

Apply:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Conceptually:

```text
Transformer Layer

q_proj
├── Base Wq ❄️
└── LoRA Aq, Bq ✓

k_proj
└── Base Wk ❄️

v_proj
├── Base Wv ❄️
└── LoRA Av, Bv ✓

o_proj
└── Base Wo ❄️
```

---

# 4. What happens if you increase LoRA rank?

This is a very important question.

Recall:

[
\Delta W = BA
]

where:

[
A \in \mathbb{R}^{r \times d_{in}}
]

[
B \in \mathbb{R}^{d_{out} \times r}
]

The rank (r) controls the complexity of the update.

---

## Example: 4096 × 4096 matrix

### Rank = 2

```text
4096
  │
  ▼
  2
  │
  ▼
4096
```

Trainable parameters:

[
2(4096+4096)
============

16,384
]

Very cheap, but limited adaptation capacity.

---

### Rank = 8

```text
4096
  │
  ▼
  8
  │
  ▼
4096
```

Trainable parameters:

[
8(8192)
=======

65,536
]

More expressive.

---

### Rank = 32

```text
4096
  │
  ▼
 32
  │
  ▼
4096
```

Trainable parameters:

[
32(8192)
========

262,144
]

Much more expressive.

---

### Rank = 128

```text
4096
  │
  ▼
128
  │
  ▼
4096
```

Trainable parameters:

[
128(8192)
=========

1,048,576
]

Much higher capacity.

---

# 5. What changes as rank increases?

## 1. More trainable parameters

The number is:

[
r(d_{in}+d_{out})
]

Therefore:

```text
r ↑
│
▼
Trainable parameters ↑
```

---

## 2. More adaptation capacity

The LoRA update has:

[
rank(BA) \leq r
]

So:

```text
Small r
↓
Simple low-dimensional update

Large r
↓
More complex update
```

This may improve performance when the adaptation task is complex.

---

## 3. More GPU memory

More trainable parameters mean more:

```text
Trainable weights
+
Gradients
+
Optimizer states
```

Therefore:

```text
r ↑
→ Memory ↑
→ Training cost ↑
→ Adapter size ↑
```

---

## 4. Higher risk of overfitting

Suppose you have only:

```text
1,000 training examples
```

Using:

```text
r = 128
```

may provide more capacity than necessary.

The model could memorize patterns rather than generalize.

So:

```text
Rank too low
    ↓
Underfitting

Rank appropriate
    ↓
Good generalization

Rank too high
    ↓
More cost / possible overfitting
```

---

# 6. Code: calculate parameter count for different ranks

```python
def lora_parameter_count(
    in_features,
    out_features,
    rank
):
    return rank * (
        in_features + out_features
    )


in_features = 4096
out_features = 4096

ranks = [2, 4, 8, 16, 32, 64, 128]

for r in ranks:

    params = lora_parameter_count(
        in_features,
        out_features,
        r
    )

    print(
        f"r={r:3} "
        f"trainable={params:,}"
    )
```

Expected output:

```text
r=  2 trainable=16,384
r=  4 trainable=32,768
r=  8 trainable=65,536
r= 16 trainable=131,072
r= 32 trainable=262,144
r= 64 trainable=524,288
r=128 trainable=1,048,576
```

Notice:

```text
Rank doubles
     ↓
LoRA parameters roughly double
```

---

# 7. How do you choose the LoRA rank?

There is no fixed answer like:

```text
Always use r = 8
```

A good engineer treats rank as a **hyperparameter**.

The best rank depends on:

1. Task complexity
2. Dataset size and diversity
3. Base model size
4. Number of target modules
5. GPU budget
6. Validation performance

---

# 8. Practical strategy

## Step 1: Start with a baseline

For many experiments:

```python
r = 8
```

or:

```python
r = 16
```

Example:

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)
```

Train and evaluate.

---

## Step 2: Test multiple ranks

For example:

```text
Experiment A → r=4
Experiment B → r=8
Experiment C → r=16
Experiment D → r=32
```

Code:

```python
ranks = [4, 8, 16, 32]

experiments = []

for r in ranks:

    config = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "v_proj"
        ],

        task_type=TaskType.CAUSAL_LM
    )

    experiments.append(config)
```

A common experiment pattern is to keep:

[
\frac{\alpha}{r}
]

roughly controlled while varying rank, though the best `alpha` should also be tuned rather than assumed.

---

# 9. Compare validation performance

Suppose results are:

| Rank | Validation Score | Trainable Params | Training Cost |
| ---: | ---------------: | ---------------: | ------------: |
|    4 |              78% |              Low |           Low |
|    8 |              84% |              Low |           Low |
|   16 |              87% |           Medium |        Medium |
|   32 |            87.5% |           Higher |        Higher |

You might choose:

```text
r = 16
```

Why?

Because:

```text
r=16 → 87%

r=32 → 87.5%
```

The additional 0.5% may not justify:

* additional training cost
* larger adapter
* more GPU memory

This is the important engineering principle:

> Choose the smallest rank that achieves the required quality.

---

# 10. Dataset size matters

## Small dataset

Example:

```text
2,000 examples
```

Start with:

```text
r = 4 or 8
```

Why?

```text
Small dataset
+
Very large adaptation capacity
=
Higher risk of overfitting
```

---

## Medium dataset

Example:

```text
20,000–100,000 examples
```

Try:

```text
r = 8, 16, 32
```

---

## Large and complex dataset

Example:

```text
Hundreds of thousands+
```

You can experiment with:

```text
r = 16, 32, 64
```

But dataset size alone does not determine the rank.

The task and model architecture also matter.

---

# 11. Task complexity matters

### Simple task

Example:

```text
Classify customer support tickets
```

Potentially:

```text
r = 4 or 8
```

---

### Moderate task

Example:

```text
Convert natural language
to structured financial JSON
```

Potentially:

```text
r = 8 or 16
```

---

### Complex task

Example:

```text
Domain-specific instruction following
+
reasoning format
+
specialized terminology
+
complex output behavior
```

Potentially test:

```text
r = 16, 32, 64
```

Again, these are experiment ranges, not guarantees.

---

# 12. Target modules and rank work together

This is very important.

Suppose you use:

```text
r = 32
```

but only target:

```text
q_proj
```

Versus:

```text
r = 8
```

but target:

```text
q_proj
k_proj
v_proj
o_proj
```

Which is better?

There is no universal answer.

Because you are changing two things:

```text
Adaptation depth
    +
Adaptation width
```

Think of:

```text
Rank
↓
How much each adapted layer can learn

Target modules
↓
How many places in the network can adapt
```

So experiment carefully.

---

# 13. A production experiment approach

Use a grid like:

```python
experiments = [
    {
        "name": "small",
        "r": 8,
        "alpha": 16,
        "targets": [
            "q_proj",
            "v_proj"
        ]
    },

    {
        "name": "medium",
        "r": 16,
        "alpha": 32,
        "targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    },

    {
        "name": "large",
        "r": 32,
        "alpha": 64,
        "targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj"
        ]
    }
]
```

Then evaluate:

```python
results = []

for experiment in experiments:

    result = {
        "name": experiment["name"],
        "validation_score": None,
        "training_time": None,
        "gpu_memory": None
    }

    # Train model here
    # Evaluate model here
    # Record metrics here

    results.append(result)
```

In a production system, you would typically log these experiments to an experiment tracker such as [MLflow](https://mlflow.org/?utm_source=chatgpt.com) or another tracking system.

---

# 14. How I would answer in an interview

## Why attention projections?

> Attention projections are commonly targeted because they are large, influential linear transformations that control how tokens attend to and exchange information. Adapting Q, K, V, and O allows the model's information flow to be modified efficiently without updating the entire model. Q and V are often used as a lightweight starting point, but the optimal target modules depend on the architecture and task.

---

## What happens when rank increases?

> Increasing LoRA rank increases the dimensionality of the low-rank update. This gives the adapter more expressive capacity because the update can have a higher rank. However, trainable parameters grow approximately linearly with rank, so GPU memory, optimizer state, training cost, adapter size, and potentially overfitting risk also increase.

---

## How do you choose rank?

> I treat rank as a hyperparameter. I start with a small baseline such as 8 or 16, evaluate on a held-out validation set, and compare against larger values such as 32 or 64. I choose the smallest rank that achieves the required quality while considering training cost, inference deployment requirements, and overfitting. I also tune rank together with target modules, alpha, learning rate, and dataset quality.

---

# Final mental model

```text
Why Attention?
────────────────────────────
Controls information flow
+
Large matrices
+
High adaptation impact


Rank r
────────────────────────────
Low r
↓
Cheap but limited

High r
↓
More expressive but expensive


Choose rank
────────────────────────────
Start small
↓
Evaluate
↓
Increase if underfitting
↓
Choose best quality/cost trade-off
```

The most important formula to remember is:

[
\boxed{
\text{LoRA parameters}
======================

r(d_{in}+d_{out})
}
]

So **rank is directly tied to both adaptation capacity and training cost**.
