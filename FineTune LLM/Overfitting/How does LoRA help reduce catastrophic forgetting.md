# How does LoRA help reduce catastrophic forgetting?

## Short answer

**LoRA helps reduce catastrophic forgetting because the original pretrained model weights remain frozen.** Instead of changing the model's learned weights directly, LoRA learns a small additional update called an **adapter**.

[
W' = W + \Delta W
]

LoRA represents the update as:

[
\Delta W = BA
]

So:

* (W) → original pretrained weights → **frozen**
* (A, B) → small LoRA matrices → **trainable**

This means the original knowledge stored in (W) is not directly overwritten.

---

# 1. First understand the catastrophic forgetting problem

Suppose we have a pretrained model:

```text
                PRETRAINED MODEL
                       │
          Knows general capabilities
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      Python        Reasoning       English
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
            Fine-tune on Insurance
                       │
                       ▼
              Specialized Model
```

With **full fine-tuning**, the model updates its original parameters:

```text
Original weight:

W

        │
        │ gradient descent
        ▼

W'

        │
        ▼

Original knowledge can change
```

The update is:

[
W_{new}
=======

## W_{old}

\eta \nabla_W L
]

Every training step modifies (W).

After thousands of updates:

```text
W
 ↓
W₁
 ↓
W₂
 ↓
W₃
 ↓
...
 ↓
W₁₀₀₀
```

Some representations that supported old capabilities may be altered.

That can cause:

```text
New domain skill       ↑
Old general skills     ↓
```

---

# 2. What changes with LoRA?

With LoRA:

```text
               PRETRAINED WEIGHT
                       W
                       │
                     FROZEN
                       │
                       │
               ┌───────┴───────┐
               │               │
               ▼               ▼
              W            LoRA A,B
           frozen           trainable
                               │
                               ▼
                            ΔW = BA
                               │
                               ▼
                        Effective Weight
```

The effective weight becomes:

[
W_{effective}
=============

W
+
BA
]

During training:

```text
W  → NOT changed
A  → changed
B  → changed
```

So the original base model is preserved.

---

# 3. Full fine-tuning vs LoRA

## Full fine-tuning

```text
Before:

W = pretrained knowledge
```

Training:

```python
W = W - learning_rate * gradient
```

After:

```text
W → changed
```

Diagram:

```text
PRETRAINED MODEL

W₁  W₂  W₃  W₄  W₅
│   │   │   │   │
▼   ▼   ▼   ▼   ▼
UPDATE ALL WEIGHTS
│   │   │   │   │
▼   ▼   ▼   ▼   ▼
W₁' W₂' W₃' W₄' W₅'
```

---

## LoRA

```text
PRETRAINED MODEL

W₁  W₂  W₃  W₄  W₅
│   │   │   │   │
F   F   F   F   F

F = Frozen


             +
             │
             ▼

      Small trainable matrices

             A × B
```

Result:

[
W_{effective}
=============

W_{frozen}
+
BA
]

The original weights stay unchanged.

---

# 4. Mathematical example

Suppose a weight matrix is:

[
W \in \mathbb{R}^{d \times d}
]

For example:

[
W \in \mathbb{R}^{4096 \times 4096}
]

Number of parameters:

[
4096 \times 4096
================

16,777,216
]

Full fine-tuning updates:

```text
16.7 million parameters
```

for this single matrix.

With LoRA, choose rank:

[
r = 8
]

Then:

[
A \in \mathbb{R}^{r \times d}
]

and:

[
B \in \mathbb{R}^{d \times r}
]

Parameter count:

[
r \times d + d \times r
]

[
8 \times 4096 + 4096 \times 8
]

[
65,536
]

So:

```text
Full fine-tuning:

16,777,216 trainable parameters


LoRA:

65,536 trainable parameters
```

The base matrix remains untouched.

---

# 5. Code: demonstrate full fine-tuning

Let's create a simple PyTorch linear layer.

```python
import torch
import torch.nn as nn
```

Create the model:

```python
model = nn.Linear(
    in_features=4,
    out_features=4
)
```

Check its parameters:

```python
for name, param in model.named_parameters():
    print(name)
    print("Requires gradient:", param.requires_grad)
```

Output:

```text
weight
Requires gradient: True

bias
Requires gradient: True
```

Now train:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

x = torch.randn(10, 4)
target = torch.randn(10, 4)

criterion = nn.MSELoss()
```

Save original weight:

```python
original_weight = model.weight.detach().clone()
```

Training step:

```python
optimizer.zero_grad()

output = model(x)

loss = criterion(
    output,
    target
)

loss.backward()

optimizer.step()
```

Now compare:

```python
weight_difference = torch.norm(
    model.weight.detach()
    -
    original_weight
)

print(weight_difference)
```

Output will be non-zero:

```text
tensor(...)
```

This proves:

```text
Original weight
      W
      │
      ▼
Training
      │
      ▼
Updated weight
      W'
```

The base knowledge itself changed.

---

# 6. Now implement a simple LoRA layer

We can implement LoRA manually.

```python
import torch
import torch.nn as nn
```

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer,
        rank=2,
        alpha=4
    ):

        super().__init__()

        self.base_layer = base_layer

        # Freeze base model
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.rank = rank
        self.alpha = alpha

        self.scaling = alpha / rank

        # A: rank × input dimension
        self.lora_A = nn.Parameter(
            torch.randn(
                rank,
                in_features
            ) * 0.01
        )

        # B: output dimension × rank
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )


    def forward(
        self,
        x
    ):

        # Original model output
        base_output = self.base_layer(x)

        # LoRA update
        lora_output = (
            x
            @ self.lora_A.T
            @ self.lora_B.T
        )

        return (
            base_output
            +
            self.scaling
            *
            lora_output
        )
```

The important part:

```python
for param in self.base_layer.parameters():
    param.requires_grad = False
```

This prevents the optimizer from changing the original model.

---

# 7. Verify which parameters are trainable

Create a base layer:

```python
base_layer = nn.Linear(
    4,
    4
)
```

Wrap it:

```python
lora_layer = LoRALinear(
    base_layer,
    rank=2,
    alpha=4
)
```

Check:

```python
for name, param in lora_layer.named_parameters():

    print(
        name,
        param.requires_grad
    )
```

Conceptually:

```text
lora_A                  True
lora_B                  True

base_layer.weight       False
base_layer.bias         False
```

This is the core reason LoRA can reduce catastrophic forgetting.

---

# 8. Train the LoRA layer

Save the original base weights:

```python
original_base_weight = (
    lora_layer
    .base_layer
    .weight
    .detach()
    .clone()
)
```

Create an optimizer containing only trainable parameters:

```python
optimizer = torch.optim.AdamW(

    filter(
        lambda parameter:
        parameter.requires_grad,

        lora_layer.parameters()
    ),

    lr=0.01
)
```

Training:

```python
x = torch.randn(
    10,
    4
)

target = torch.randn(
    10,
    4
)

criterion = nn.MSELoss()


for step in range(100):

    optimizer.zero_grad()

    output = lora_layer(x)

    loss = criterion(
        output,
        target
    )

    loss.backward()

    optimizer.step()
```

Now check whether the base weights changed:

```python
base_weight_difference = torch.norm(

    lora_layer
    .base_layer
    .weight
    .detach()

    -

    original_base_weight
)

print(
    "Base weight difference:",
    base_weight_difference.item()
)
```

Expected result:

```text
Base weight difference: 0.0
```

But check the LoRA matrices:

```python
print(
    torch.norm(
        lora_layer.lora_A
    )
)

print(
    torch.norm(
        lora_layer.lora_B
    )
)
```

They changed during training.

So:

```text
BASE MODEL WEIGHTS

W

Before training:
████████████████

After training:
████████████████

Same
```

But:

```text
LoRA ADAPTER

Before:
small / near zero

After:
learned adaptation
```

---

# 9. Why this reduces forgetting

Imagine the original model knows:

```text
Python
Reasoning
General language
Summarization
Math
```

These capabilities are encoded across the base weights:

[
W
]

With full fine-tuning:

```text
W → W'
```

The original representation itself is modified.

With LoRA:

```text
W → frozen

New behavior:

W + ΔW
```

where:

[
\Delta W = BA
]

The base knowledge remains available in:

[
W
]

The adapter adds specialized behavior:

```text
              Original capability
                     │
                     ▼
                    W
                 frozen
                     │
                     +
                     │
                     ▼
             Domain adaptation
                   ΔW
                     │
                     ▼
              W + ΔW output
```

---

# 10. Important point: LoRA does NOT completely eliminate forgetting

This is extremely important in an interview.

Even though:

```text
W remains unchanged
```

the model's **effective behavior** is:

[
W_{effective}
=============

W
+
\Delta W
]

If:

[
\Delta W
]

is large enough, the final output can change significantly.

Example:

```text
Base model:

"What is Python?"

→ Python is a programming language.


Base + LoRA adapter:

"What is Python?"

→ Please contact our insurance support team.
```

This is possible if the adapter was trained badly or too aggressively.

So the correct statement is:

> **LoRA reduces the risk of catastrophic forgetting at the parameter level because it does not overwrite the base weights, but it does not guarantee preservation of behavior when the adapter is active.**

---

# 11. How to make LoRA safer

## A. Use a small rank

Example:

```python
lora_config = LoraConfig(
    r=8
)
```

A smaller rank limits the adaptation capacity.

Conceptually:

```text
Small r:

Base knowledge
██████████████████

Adapter influence
██
```

Higher rank:

```text
Base knowledge
██████████████████

Adapter influence
██████████
```

A larger rank can learn more complex changes, which is useful, but may also allow stronger behavioral deviation.

---

# 12. Use reasonable alpha/scaling

LoRA scaling is:

[
Scaling
=======

\frac{\alpha}{r}
]

Example:

```python
r = 8
alpha = 16
```

Then:

[
Scaling
=======

# \frac{16}{8}

2
]

Code:

```python
config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05
)
```

The effective update is:

[
W' =
W
+
\frac{\alpha}{r}BA
]

If scaling is too large:

```text
ΔW becomes more influential
```

and the adapter can strongly override base behavior.

---

# 13. Use LoRA dropout

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type="CAUSAL_LM"
)
```

Dropout regularizes adapter training.

This can reduce overfitting to a narrow dataset.

---

# 14. Use fewer target modules

Instead of adapting everything:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

you may start with:

```python
target_modules = [
    "q_proj",
    "v_proj"
]
```

This gives the adapter fewer places to modify the model's computation.

Conceptually:

```text
Adapt everything:

Attention   ███████
MLP         ███████
Output      ███████


Adapt selected layers:

Q           ███████
V           ███████

Others      Frozen
```

The best target modules depend on the model architecture and task.

---

# 15. Mix general data with domain data

LoRA protects the parameters, but **data mixing protects behavior**.

Example:

```text
Domain data:

80%

General/replay data:

20%
```

Code:

```python
from datasets import concatenate_datasets


domain_dataset = domain_dataset.map(
    lambda x: {
        "source": "domain"
    }
)

general_dataset = general_dataset.map(
    lambda x: {
        "source": "general"
    }
)


training_dataset = concatenate_datasets([

    domain_dataset,

    general_dataset
]).shuffle(seed=42)
```

For a real ratio:

```python
domain_sample = domain_dataset.shuffle(
    seed=42
).select(
    range(8000)
)

general_sample = general_dataset.shuffle(
    seed=42
).select(
    range(2000)
)
```

Then:

```python
training_dataset = concatenate_datasets([
    domain_sample,
    general_sample
]).shuffle(seed=42)
```

This approximates:

```text
80% domain
20% general
```

---

# 16. Full practical LoRA example

Here is a more realistic Hugging Face + PEFT setup.

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)
```

Load model:

```python
MODEL_NAME = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Set padding:

```python
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token
```

Create LoRA configuration:

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

Apply LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check:

```python
model.print_trainable_parameters()
```

Conceptually:

```text
trainable params: 0.1% - 2%
all params: 100%
```

The exact percentage depends on:

* model size
* architecture
* target modules
* LoRA rank

---

# 17. Verify that base weights are frozen

```python
for name, parameter in model.named_parameters():

    if "lora_" not in name:

        if parameter.requires_grad:

            print(
                "Unexpected trainable parameter:",
                name
            )
```

Check adapter parameters:

```python
for name, parameter in model.named_parameters():

    if "lora_" in name:

        print(
            name,
            parameter.requires_grad
        )
```

Conceptually:

```text
Base model:
requires_grad = False

LoRA parameters:
requires_grad = True
```

---

# 18. Conservative training configuration

```python
training_args = TrainingArguments(

    output_dir="./lora_customer_support",

    num_train_epochs=3,

    learning_rate=1e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    warmup_ratio=0.05,

    weight_decay=0.01,

    lr_scheduler_type="cosine",

    bf16=True,

    logging_steps=10,

    save_strategy="epoch",

    eval_strategy="epoch",

    load_best_model_at_end=True,

    report_to="none"
)
```

Train:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)
```

```python
trainer.train()
```

---

# 19. Compare full fine-tuning and LoRA

| Feature                            | Full Fine-Tuning | LoRA                    |
| ---------------------------------- | ---------------- | ----------------------- |
| Base weights                       | Updated          | Frozen                  |
| Trainable parameters               | Very high        | Low                     |
| Memory                             | High             | Lower                   |
| Original weights overwritten       | Yes              | No                      |
| Risk of parameter-level forgetting | Higher           | Lower                   |
| Can adapted behavior change?       | Yes              | Yes                     |
| Can fully eliminate forgetting?    | No               | No                      |
| Easy rollback                      | Harder           | Easy—disable adapter    |
| Multiple domain versions           | Expensive        | Multiple small adapters |

---

# 20. A very important advantage: disable the adapter

With full fine-tuning:

```text
Base model
     │
     ▼
Weights modified permanently
```

Recovering original behavior requires keeping a separate original checkpoint.

With LoRA:

```text
Base model
     │
     ├── Insurance adapter
     │
     ├── Legal adapter
     │
     └── Medical adapter
```

Conceptually:

```python
# Domain behavior
output = model_with_lora(prompt)
```

Disable the adapter:

```python
with model.disable_adapter():
    output = model(prompt)
```

Now you return to the original base-model behavior because the underlying base weights were not changed.

This is a major operational advantage.

---

# 21. LoRA + replay is better than LoRA alone

My preferred strategy is:

```text
                   BASE MODEL
                        │
                     Frozen
                        │
                        +
                        │
                 LoRA adapters
                        │
                        ▼
              Domain adaptation
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼

    Domain examples              General replay
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
                   Mixed loss
                        │
                        ▼
                 Train adapters
```

Mathematically:

[
L =
\lambda L_{domain}
+
(1-\lambda)L_{general}
]

For example:

[
L =
0.8L_{domain}
+
0.2L_{general}
]

This provides a strong balance:

```text
LoRA:
Protects base parameters

Replay:
Preserves behavioral capabilities

Evaluation:
Detects regressions
```

---

# 22. Production evaluation for forgetting

Before fine-tuning:

```python
baseline_scores = {
    "general_qa": 0.90,
    "reasoning": 0.85,
    "coding": 0.88,
    "domain": 0.65
}
```

After LoRA:

```python
lora_scores = {
    "general_qa": 0.89,
    "reasoning": 0.84,
    "coding": 0.87,
    "domain": 0.94
}
```

Calculate:

```python
MAX_ALLOWED_DROP = 0.05


for capability in baseline_scores:

    before = baseline_scores[capability]

    after = lora_scores[capability]

    change = after - before

    print(
        f"{capability}: "
        f"{change:+.2%}"
    )

    if (
        capability != "domain"
        and change < -MAX_ALLOWED_DROP
    ):

        print(
            "WARNING: capability regression"
        )
```

Possible output:

```text
general_qa: +1.00%
reasoning: -1.00%
coding: -1.00%
domain: +29.00%
```

This is much more useful than saying:

```text
Training loss decreased
```

because low training loss does **not** prove that general capabilities were preserved.

---

# Best interview answer

> **LoRA helps reduce catastrophic forgetting primarily because it freezes the pretrained base model and learns a low-rank update through small trainable matrices. Instead of replacing the original weight (W), LoRA computes an effective weight (W + \Delta W), where (\Delta W = BA). Therefore, the pretrained parameters and their original knowledge are not directly overwritten.**
>
> **However, LoRA does not completely eliminate behavioral forgetting because the adapter can still significantly change the model's output when active. In production, I would combine LoRA with replay data, conservative rank and scaling, appropriate learning rates, early stopping, and regression evaluations on both the new domain and the original capabilities. A major operational advantage is that the adapter can be disabled or swapped, immediately returning to the untouched base model.**

## The key mental model

```text
FULL FINE-TUNING

W
↓
modify W
↓
W'
↓
Old knowledge may be overwritten


LoRA

W ─────────────── Frozen
│
│
└─────── + ΔW (trainable adapter)
              │
              ▼
          New capability


LoRA + Replay

Frozen W
   +
LoRA ΔW
   +
General replay data
   +
Domain data
   +
Regression evaluation
   │
   ▼
Better preservation of general capabilities
```

**One sentence to remember:**

> **Full fine-tuning changes the knowledge; LoRA keeps the base knowledge intact and learns an additional adaptation.**
