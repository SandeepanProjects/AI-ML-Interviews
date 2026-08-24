# LoRA (Low-Rank Adaptation) — explained properly

## 1. What problem does LoRA solve?

Suppose you have a large pre-trained LLM and want to adapt it to a new domain.

With **full fine-tuning**, you update every weight:

```text
Pretrained Model

W1  ✓ train
W2  ✓ train
W3  ✓ train
W4  ✓ train
...
Billions of parameters updated
```

This is expensive because training requires memory for:

* model weights
* gradients
* optimizer states
* activations

LoRA asks:

> Do we really need to update every parameter in a large weight matrix?

Instead, LoRA:

1. **Freezes the original weight matrix**
2. Learns a small update to that matrix
3. Represents that update using two small low-rank matrices

---

# 2. The core idea of LoRA

Suppose a neural network layer has a weight matrix:

[
W_0
]

During full fine-tuning:

[
W = W_0 + \Delta W
]

Where:

* (W_0) = original pre-trained weights
* (\Delta W) = learned update

Full fine-tuning learns every element of:

[
\Delta W
]

LoRA approximates that update as:

[
\Delta W = BA
]

Therefore:

[
W = W_0 + BA
]

Usually:

[
W_0
]

is frozen.

Only:

[
A \quad \text{and} \quad B
]

are trained.

---

# 3. Visual intuition

Suppose the original matrix is:

```text
W₀

4096 × 4096
```

Full fine-tuning:

```text
4096 × 4096
     ↓
16,777,216 trainable parameters
```

LoRA chooses a small rank, for example:

```text
r = 8
```

Then:

```text
A

8 × 4096
```

and:

```text
B

4096 × 8
```

So instead of training:

[
4096 \times 4096
================

16,777,216
]

parameters, LoRA trains:

[
(8 \times 4096) + (4096 \times 8)
]

[
= 65,536
]

This is:

```text
Full Fine-Tuning:

████████████████████████
16,777,216 parameters


LoRA:

█
65,536 parameters
```

---

# 4. Why is it called Low-Rank Adaptation?

The rank of a matrix determines, roughly, how much independent information the matrix can represent.

A full matrix:

[
\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}
]

can have rank up to:

[
\min(d_{out}, d_{in})
]

For:

```text
4096 × 4096
```

the maximum rank is:

```text
4096
```

LoRA approximates the update with:

[
\Delta W = BA
]

where:

[
B \in \mathbb{R}^{d_{out} \times r}
]

and:

[
A \in \mathbb{R}^{r \times d_{in}}
]

If:

```text
r = 8
```

then:

[
rank(BA) \leq 8
]

So LoRA assumes:

> The important adaptation needed for the new task can often be represented in a much lower-dimensional subspace.

That is why it is called **Low-Rank Adaptation**.

---

# 5. LoRA mathematics

Suppose a normal linear layer is:

[
Y = XW_0
]

With LoRA:

[
Y = X(W_0 + \Delta W)
]

LoRA:

[
\Delta W = BA
]

Therefore:

[
Y = X(W_0 + BA)
]

Usually LoRA also uses scaling:

[
Y = XW_0 + XBA \times \frac{\alpha}{r}
]

Where:

| Symbol   | Meaning                   |
| -------- | ------------------------- |
| (W_0)    | Frozen original weight    |
| (A)      | Trainable low-rank matrix |
| (B)      | Trainable low-rank matrix |
| (r)      | Rank                      |
| (\alpha) | Scaling factor            |

---

# 6. Matrix dimensions

Suppose:

```text
Input dimension = 4096
Output dimension = 4096
Rank = 8
```

Original weight:

```text
W₀

4096 × 4096
```

LoRA:

```text
A

4096 × 8
```

and:

```text
B

8 × 4096
```

Depending on matrix multiplication convention, libraries may define the orientation differently. The important relationship is:

```text
ΔW = B × A
```

with dimensions:

```text
B: d_out × r
A: r × d_in
```

So:

```text
(d_out × r) × (r × d_in)

        ↓

d_out × d_in
```

---

# 7. Why does LoRA reduce trainable parameters?

Let's compare mathematically.

## Full fine-tuning

A linear layer:

[
W \in \mathbb{R}^{d_{out} \times d_{in}}
]

Number of trainable parameters:

[
d_{out} \times d_{in}
]

---

## LoRA

LoRA matrices:

[
A \in \mathbb{R}^{r \times d_{in}}
]

[
B \in \mathbb{R}^{d_{out} \times r}
]

Trainable parameters:

[
r \times d_{in}
+
d_{out} \times r
]

or:

[
r(d_{in} + d_{out})
]

---

## Example

For:

```text
d_in = 4096
d_out = 4096
r = 8
```

### Full fine-tuning

[
4096 \times 4096 = 16,777,216
]

### LoRA

[
8(4096 + 4096)
]

[
8 \times 8192 = 65,536
]

Reduction:

[
\frac{65,536}{16,777,216}
\approx 0.39%
]

So LoRA trains only about:

```text
0.39%
```

of the parameters for that layer.

---

# 8. LoRA implementation from scratch

Let's build a LoRA linear layer using PyTorch.

```python
import torch
import torch.nn as nn
import math
```

## Create a LoRA layer

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # ---------------------------------
        # Original pretrained weight
        # ---------------------------------

        self.weight = nn.Parameter(
            torch.randn(
                out_features,
                in_features
            )
        )

        # Freeze base weight
        self.weight.requires_grad = False

        # ---------------------------------
        # LoRA matrices
        # ---------------------------------

        # A: rank × input_dimension
        self.A = nn.Parameter(
            torch.randn(
                rank,
                in_features
            )
        )

        # B: output_dimension × rank
        self.B = nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )

        # Scaling
        self.scaling = alpha / rank
```

### Why initialize `B` with zeros?

Initially:

[
BA = 0
]

So:

[
W = W_0
]

This means training starts with the original pre-trained model behavior.

---

## Forward pass

```python
    def forward(self, x):

        # Original frozen model output
        base_output = x @ self.weight.T

        # LoRA update
        lora_update = (
            (x @ self.A.T)
            @ self.B.T
        )

        # Add scaled update
        return (
            base_output
            +
            self.scaling * lora_update
        )
```

Complete class:

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        alpha=16
    ):
        super().__init__()

        # Original pretrained weight
        self.weight = nn.Parameter(
            torch.randn(
                out_features,
                in_features
            )
        )

        # Freeze base model weight
        self.weight.requires_grad = False

        # LoRA A
        self.A = nn.Parameter(
            torch.randn(
                rank,
                in_features
            ) * 0.01
        )

        # LoRA B
        self.B = nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )

        # Scaling
        self.scaling = alpha / rank


    def forward(self, x):

        # Base model
        base_output = (
            x @ self.weight.T
        )

        # LoRA adaptation
        lora_output = (
            x
            @ self.A.T
            @ self.B.T
        )

        return (
            base_output
            +
            self.scaling
            * lora_output
        )
```

---

# 9. Test the LoRA layer

```python
lora_layer = LoRALinear(
    in_features=128,
    out_features=256,
    rank=8,
    alpha=16
)

x = torch.randn(
    32,
    128
)

output = lora_layer(x)

print(output.shape)
```

Output:

```text
torch.Size([32, 256])
```

---

# 10. Check which parameters are trainable

```python
for name, parameter in lora_layer.named_parameters():

    print(
        name,
        parameter.shape,
        parameter.requires_grad
    )
```

Conceptually:

```text
weight     [256, 128]   False ❄️

A          [8, 128]     True ✓

B          [256, 8]     True ✓
```

So:

```text
Base Weight
     ❄️ Frozen

LoRA A
     ✓ Train

LoRA B
     ✓ Train
```

---

# 11. Count parameters with code

```python
def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(
        f"Trainable percentage: "
        f"{100 * trainable / total:.2f}%"
    )
```

Run:

```python
count_parameters(lora_layer)
```

For:

```text
Input = 128
Output = 256
Rank = 8
```

Full weight:

[
128 \times 256 = 32,768
]

LoRA:

[
(8 \times 128)
+
(256 \times 8)
]

[
1,024 + 2,048
=============

3,072
]

So instead of updating:

```text
32,768 parameters
```

you update:

```text
3,072 parameters
```

---

# 12. Compare Full Fine-Tuning vs LoRA in code

## Full fine-tuning layer

```python
class FullFineTuningLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x):

        return self.linear(x)
```

Every parameter is trainable.

```python
full_model = FullFineTuningLinear(
    4096,
    4096
)

count_parameters(full_model)
```

Conceptually:

```text
Total: ~16.7 million
Trainable: ~16.7 million
```

---

## LoRA layer

```python
lora_model = LoRALinear(
    in_features=4096,
    out_features=4096,
    rank=8
)

count_parameters(lora_model)
```

Conceptually:

```text
Total:
16,842,752

Trainable:
65,536
```

The original weight still exists in memory, but it is frozen.

This distinction is important:

> **LoRA reduces the number of trainable parameters, gradients, and optimizer states. It does not magically remove the memory required to hold the base model.**

---

# 13. Training the LoRA layer

```python
from torch.optim import AdamW
```

Create an optimizer:

```python
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        lora_layer.parameters()
    ),
    lr=1e-3
)
```

Training loop:

```python
loss_function = nn.MSELoss()

for epoch in range(10):

    # Input
    x = torch.randn(
        32,
        128
    )

    # Target
    target = torch.randn(
        32,
        256
    )

    # Forward
    output = lora_layer(x)

    # Loss
    loss = loss_function(
        output,
        target
    )

    # Clear gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update only A and B
    optimizer.step()

    print(
        f"Epoch {epoch}: "
        f"{loss.item():.4f}"
    )
```

Internally:

```text
Loss
 │
 ▼
Backpropagation
 │
 ├── Frozen W₀ ❄️
 │       No update
 │
 ├── LoRA A ✓
 │       Gradient
 │
 └── LoRA B ✓
         Gradient
```

---

# 14. Verify that only LoRA weights change

```python
# Save copies before training
base_before = (
    lora_layer.weight
    .detach()
    .clone()
)

A_before = (
    lora_layer.A
    .detach()
    .clone()
)

B_before = (
    lora_layer.B
    .detach()
    .clone()
)
```

After one training step:

```python
base_changed = not torch.equal(
    base_before,
    lora_layer.weight
)

A_changed = not torch.equal(
    A_before,
    lora_layer.A
)

B_changed = not torch.equal(
    B_before,
    lora_layer.B
)

print("Base changed:", base_changed)
print("A changed:", A_changed)
print("B changed:", B_changed)
```

Expected:

```text
Base changed: False
A changed: True
B changed: True
```

This demonstrates the fundamental LoRA idea.

---

# 15. Where is LoRA applied in an LLM?

A Transformer contains several large linear projections.

For attention:

```text
Input
 │
 ├── Wq → Query
 │
 ├── Wk → Key
 │
 ├── Wv → Value
 │
 └── Wo → Output
```

LoRA can be applied to selected projections:

```text
Wq → Frozen + LoRA
Wk → Frozen
Wv → Frozen + LoRA
Wo → Frozen
```

Or more broadly:

```text
Attention projections
      +
MLP projections
```

Conceptually:

```text
Original:

Q = XWq

LoRA:

Q = X(Wq + BqAq)
```

Similarly:

[
V = X(W_v + B_vA_v)
]

---

# 16. Real Hugging Face LoRA example

Install:

```bash
pip install torch transformers peft datasets accelerate
```

Load a model:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Import PEFT:

```python
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
```

Configure LoRA:

```python
config = LoraConfig(

    # Low-rank dimension
    r=8,

    # Scaling factor
    lora_alpha=16,

    # Regularization
    lora_dropout=0.05,

    # Apply LoRA to attention projection
    target_modules=[
        "c_attn"
    ],

    # Don't train original biases
    bias="none",

    task_type=TaskType.CAUSAL_LM
)
```

Apply LoRA:

```python
lora_model = get_peft_model(
    model,
    config
)
```

Check:

```python
lora_model.print_trainable_parameters()
```

You should see something conceptually like:

```text
trainable params:
small percentage

all params:
124M

trainable %:
much smaller than 100%
```

The exact number depends on:

* model architecture
* target modules
* rank
* PEFT version/configuration

---

# 17. Preparing a small dataset

Example data:

```python
from datasets import Dataset

data = {
    "text": [
        "Python is a programming language.",
        "RAG combines retrieval with generation.",
        "LoRA is a parameter efficient fine tuning method."
    ]
}

dataset = Dataset.from_dict(data)
```

For causal language modeling:

```python
def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )
```

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)
```

For actual causal-LM training, you also need labels. A common setup is:

```python
def tokenize_function(examples):

    output = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

    output["labels"] = (
        output["input_ids"].copy()
    )

    return output
```

---

# 18. Train LoRA

```python
from transformers import (
    Trainer,
    TrainingArguments
)
```

```python
training_args = TrainingArguments(

    output_dir="./lora_output",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none"
)
```

Create trainer:

```python
trainer = Trainer(

    model=lora_model,

    args=training_args,

    train_dataset=tokenized_dataset
)
```

Train:

```python
trainer.train()
```

During training:

```text
Base GPT-2 weights
       ❄️
       │
       │ frozen
       ▼

LoRA A/B
       ✓
       │
       ▼

   Backpropagation
       │
       ▼

   Optimizer updates
```

---

# 19. Saving the LoRA adapter

```python
lora_model.save_pretrained(
    "./my_lora_adapter"
)
```

Usually this saves primarily the adapter configuration and adapter weights, not a separate full copy of the entire base model.

Later:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = PeftModel.from_pretrained(
    base_model,
    "./my_lora_adapter"
)
```

Conceptually:

```text
Base Model
     +
LoRA Adapter
     │
     ▼
Specialized Model
```

---

# 20. LoRA merging

Because:

[
W = W_0 + BA
]

LoRA updates can often be merged into the original weight:

```python
merged_model = lora_model.merge_and_unload()
```

Conceptually:

```text
Before:

W₀ + BA


After merging:

Wfinal
```

This can simplify inference for compatible configurations.

---

# 21. Rank `r` — an important interview topic

The LoRA rank controls the size of the adaptation.

```text
r = 2
Very small
Less expressive

r = 8
Common starting point

r = 16
More capacity

r = 64
More adaptation capacity
More trainable parameters
```

Example:

```text
4096 × 4096 layer

Rank 4:
32,768 trainable parameters

Rank 8:
65,536

Rank 16:
131,072

Rank 64:
524,288
```

Increasing rank:

```text
More capacity
      ↑
More parameters
      ↑
More memory
      ↑
Potentially better adaptation
```

But bigger is not always better.

---

# 22. What does `lora_alpha` do?

The LoRA update is usually scaled:

[
\Delta W
========

\frac{\alpha}{r}BA
]

Example:

```python
r = 8
alpha = 16
```

Scaling:

```python
16 / 8 = 2
```

So:

```text
Final update = 2 × BA
```

`alpha` controls the effective strength of the LoRA update.

---

# 23. Why LoRA is efficient in practice

LoRA reduces:

### 1. Trainable parameters

```text
Full FT:
Billions trainable

LoRA:
Millions or fewer trainable
```

### 2. Gradient memory

```text
Full FT:
Gradient for almost every parameter

LoRA:
Gradients mainly for A and B
```

### 3. Optimizer memory

Optimizers such as Adam/AdamW maintain additional state.

```text
Full FT:

Weights
Gradients
Adam moment 1
Adam moment 2
```

for nearly every trainable parameter.

With LoRA:

```text
Frozen base model:
No optimizer states required

LoRA parameters:
Optimizer states required
```

This can significantly reduce training memory.

---

# 24. LoRA vs Full Fine-Tuning

| Feature               | Full Fine-Tuning      | LoRA                          |
| --------------------- | --------------------- | ----------------------------- |
| Base weights          | Updated               | Frozen                        |
| Trainable parameters  | Nearly all            | Small subset                  |
| Extra matrices        | No                    | A and B                       |
| GPU memory            | High                  | Lower                         |
| Training cost         | High                  | Lower                         |
| Storage per task      | Full model            | Small adapter                 |
| Multiple domains      | Expensive             | Efficient                     |
| Merge into base model | Already part of model | Often possible                |
| Adaptation capacity   | Maximum               | Depends on rank/configuration |

---

# 25. Simple analogy

Imagine a huge textbook with 10,000 pages.

## Full fine-tuning

You rewrite every page:

```text
Page 1  ✏️
Page 2  ✏️
Page 3  ✏️
...
Page 10000 ✏️
```

## LoRA

You keep the textbook unchanged and add a small notebook containing only the important modifications:

```text
Original Textbook ❄️

+
Small Modification Notebook ✓
```

The modification notebook is:

```text
A and B
```

---

# Best interview answer

> **LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning technique where the original pre-trained model weights are frozen. Instead of learning a full weight update for a large matrix (W), LoRA represents the update as the product of two smaller matrices, typically (\Delta W = BA), where the rank (r) is much smaller than the original dimensions.**
>
> **For example, instead of training a 4096 × 4096 matrix with about 16.7 million parameters, a rank-8 LoRA update trains matrices with only (8(4096 + 4096) = 65,536) parameters. This reduces trainable parameters, gradient memory, optimizer-state memory, and storage for task-specific adaptations while preserving the original base model.**

## One-line memory trick

```text
Full Fine-Tuning:
Train W

LoRA:
Freeze W
Train ΔW = BA

where:
r << model dimension
```

That is the core reason **LoRA dramatically reduces the number of trainable parameters**.
