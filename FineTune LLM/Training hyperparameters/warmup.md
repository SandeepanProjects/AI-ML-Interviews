# 1. What is Warmup?

**Warmup** is a learning-rate strategy where we start training with a **small learning rate** and gradually increase it to the target learning rate during the initial part of training.

Instead of this:

```text
Step 1:     LR = 0.0001
Step 2:     LR = 0.0001
Step 3:     LR = 0.0001
```

we do:

```text
Step 1:     LR = 0.00001
Step 2:     LR = 0.00003
Step 3:     LR = 0.00006
Step 4:     LR = 0.00008
Step 5:     LR = 0.00010
```

After reaching the target learning rate, a scheduler may keep it constant or decay it.

---

# 2. Why do we need warmup for LLM fine-tuning?

At the beginning of training:

* model parameters may receive unstable gradients
* optimizer states, especially Adam/AdamW momentum statistics, are not yet well established
* a large learning rate immediately can cause large parameter updates
* mixed precision training can make instability more noticeable

Imagine your target learning rate is:

```python
learning_rate = 1e-4
```

Without warmup:

```text
Step 1
  ↓
LR = 1e-4
  ↓
Large parameter update immediately
```

With warmup:

```text
Step 1  → LR = 1e-5
Step 2  → LR = 2e-5
Step 3  → LR = 4e-5
Step 4  → LR = 7e-5
Step 5  → LR = 1e-4
```

This allows training to start more gradually.

---

# 3. Warmup visualization

Without warmup:

```text
Learning Rate

1e-4 ─────────────────────────────
      │
      │
0     └───────────────────────────
      Training steps
```

With warmup:

```text
Learning Rate

1e-4              ───────────────
                 /
               /
             /
           /
0 ───────/────────────────────────
        Warmup
```

With warmup + decay:

```text
Learning Rate

1e-4             /\
                /  \
               /    \
              /      \
             /        \
0 ──────────/──────────\──────────
           Warmup        Decay
```

---

# 4. Linear warmup mathematically

Suppose:

```python
target_lr = 1e-4
warmup_steps = 1000
```

A simple linear warmup is:

[
LR_t =
LR_{target}
\times
\frac{t}{warmup_steps}
]

for:

[
t \leq warmup_steps
]

Example:

```text
Target LR = 0.0001
Warmup steps = 1000
```

At step 100:

[
0.0001 \times \frac{100}{1000}
==============================

0.00001
]

At step 500:

[
0.0001 \times \frac{500}{1000}
==============================

0.00005
]

At step 1000:

[
0.0001 \times 1
===============

0.0001
]

---

# 5. Code: implement linear warmup manually

```python
target_lr = 1e-4
warmup_steps = 1000


def get_warmup_lr(step: int) -> float:
    if step >= warmup_steps:
        return target_lr

    return (
        target_lr
        * step
        / warmup_steps
    )


for step in [0, 100, 500, 1000, 1500]:
    print(
        f"Step {step}: "
        f"LR = {get_warmup_lr(step):.8f}"
    )
```

Output approximately:

```text
Step 0:    LR = 0.00000000
Step 100:  LR = 0.00001000
Step 500:  LR = 0.00005000
Step 1000: LR = 0.00010000
Step 1500: LR = 0.00010000
```

In real training, you usually don't implement this manually.

---

# 6. Warmup using PyTorch scheduler

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(
    model.parameters(),
    lr=1e-4
)

num_training_steps = 10_000
num_warmup_steps = 500

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

Training loop:

```python
for batch in train_dataloader:

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    # Update learning rate
    scheduler.step()
```

The typical order is:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer step
   ↓
Scheduler step
```

---

# 7. Warmup with gradient accumulation

This is important for LLMs.

Suppose:

```python
gradient_accumulation_steps = 4
```

You should generally update the optimizer and learning-rate scheduler when an actual optimizer step occurs.

Example:

```python
optimizer.zero_grad()

for step, batch in enumerate(train_dataloader):

    outputs = model(**batch)

    loss = (
        outputs.loss
        / gradient_accumulation_steps
    )

    loss.backward()

    if (
        (step + 1)
        % gradient_accumulation_steps
        == 0
    ):

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()
```

Conceptually:

```text
Micro-batch 1
    ↓
Accumulate gradient

Micro-batch 2
    ↓
Accumulate gradient

Micro-batch 3
    ↓
Accumulate gradient

Micro-batch 4
    ↓
optimizer.step()
scheduler.step()
```

Warmup steps usually refer to **optimizer updates**, not every micro-batch.

---

# 8. `warmup_steps` vs `warmup_ratio`

## `warmup_steps`

You explicitly specify:

```python
warmup_steps = 500
```

Meaning:

> Gradually increase the LR during the first 500 optimizer steps.

---

## `warmup_ratio`

You specify a percentage:

```python
warmup_ratio = 0.05
```

Meaning:

> Use 5% of the total training steps for warmup.

Example:

```text
Total training steps = 10,000

Warmup ratio = 0.05

Warmup steps = 500
```

Code:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    num_train_epochs=3
)
```

For most experiments, `warmup_ratio` is convenient because if the dataset size changes, the warmup duration scales automatically.

---

# 9. How much warmup should you use?

There is no universal answer.

Reasonable starting experiments might be:

```text
0%      → no warmup
1%      → small warmup
3%      → moderate warmup
5%      → common starting point
10%     → relatively long warmup
```

For example:

```python
warmup_ratio = 0.03
```

or:

```python
warmup_ratio = 0.05
```

I would tune it based on:

```text
Dataset size
Total training steps
Learning rate
Base model
Fine-tuning method
Training stability
Batch size
```

---

# 10. Warmup + cosine learning-rate decay

A common LLM setup is:

```text
Warmup
   ↓
Reach maximum LR
   ↓
Gradually decrease LR
```

Conceptually:

```text
Learning Rate

                Maximum LR
                     /\
                    /  \
                   /    \
                  /      \
                 /        \
________________/          \____

      Warmup       Cosine decay
```

Using Hugging Face:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    num_train_epochs=3
)
```

---

# 11. What is Weight Decay?

**Weight decay** is a regularization technique that discourages model weights from becoming excessively large.

Conceptually:

[
Loss_{total}
============

Loss_{task}
+
\lambda \cdot Penalty(weights)
]

For classical L2 regularization:

[
Loss_{total}
============

Loss_{task}
+
\lambda \sum_i w_i^2
]

where:

* (w_i) = model parameters
* (\lambda) = regularization strength

The purpose is to reduce overfitting and encourage simpler parameter values.

---

# 12. Intuition behind weight decay

Suppose the model has a weight:

```text
w = 10.0
```

The optimizer updates it based on the training gradient.

Without weight decay:

```text
w = w - learning_rate × gradient
```

With weight decay, parameters are also nudged toward smaller values.

A simplified conceptual form:

[
w
\leftarrow
(1 - lr \times wd)w
-------------------

lr \times gradient
]

So:

```text
Large weights
    ↓
Regularization pressure
    ↓
Smaller weights
```

This can help reduce overfitting.

---

# 13. Simple weight decay example

Suppose:

```python
weight = 10.0
learning_rate = 0.1
weight_decay = 0.01
```

Ignoring the task gradient for a moment:

```python
new_weight = (
    weight
    * (1 - learning_rate * weight_decay)
)

print(new_weight)
```

Output:

```text
9.99
```

Repeated updates gradually shrink the weight.

Example:

```python
weight = 10.0

learning_rate = 0.1
weight_decay = 0.01

for step in range(10):

    weight = (
        weight
        * (1 - learning_rate * weight_decay)
    )

    print(
        f"Step {step + 1}: "
        f"{weight:.4f}"
    )
```

Output approximately:

```text
Step 1: 9.9900
Step 2: 9.9800
Step 3: 9.9700
...
```

In real optimization, task gradients and optimizer behavior also affect the parameter.

---

# 14. L2 regularization vs weight decay

This distinction is important.

People often use these terms interchangeably, but with adaptive optimizers such as Adam, they are not exactly the same implementation.

## L2 regularization

You add a penalty to the loss:

[
Loss
====

Loss_{task}
+
\lambda \sum w^2
]

Example:

```python
loss = task_loss

l2_penalty = 0

for parameter in model.parameters():

    l2_penalty += (
        parameter.pow(2).sum()
    )

loss = (
    task_loss
    + 0.01 * l2_penalty
)
```

Then:

```python
loss.backward()
optimizer.step()
```

---

## AdamW weight decay

AdamW decouples weight decay from the adaptive gradient update.

Instead of treating the penalty purely as part of the loss gradient, it applies a separate decay to parameters.

This is why, for transformer/LLM training, you commonly see:

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

For modern transformer fine-tuning, **AdamW is a very common optimizer choice**.

---

# 15. Code: weight decay with PyTorch

```python
import torch

model = torch.nn.Linear(
    in_features=10,
    out_features=1
)

optimizer = torch.optim.AdamW(
    model.parameters(),

    lr=1e-3,

    weight_decay=0.01
)
```

Training:

```python
loss_fn = torch.nn.MSELoss()

for inputs, targets in dataloader:

    optimizer.zero_grad()

    predictions = model(inputs)

    loss = loss_fn(
        predictions,
        targets
    )

    loss.backward()

    optimizer.step()
```

You don't need to manually add the regularization term.

The optimizer handles weight decay.

---

# 16. Should weight decay be applied to every parameter?

Usually, **no**.

A common Transformer setup excludes parameters such as:

* bias terms
* normalization parameters

Why?

Normalization parameters and biases often should not be regularized in the same way as large weight matrices.

Example parameter grouping:

```python
from torch.optim import AdamW

decay_parameters = []
no_decay_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if (
        "bias" in name
        or "LayerNorm.weight" in name
        or "layer_norm.weight" in name
    ):
        no_decay_parameters.append(parameter)

    else:
        decay_parameters.append(parameter)


optimizer = AdamW(
    [
        {
            "params": decay_parameters,
            "weight_decay": 0.01
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0
        }
    ],
    lr=1e-4
)
```

The idea is:

```text
Large weight matrices
        ↓
Apply decay

Bias / normalization parameters
        ↓
Usually no decay
```

In practice, the exact parameter grouping should be verified for the specific model architecture and training framework.

---

# 17. Weight decay during LoRA fine-tuning

Suppose your base model is frozen:

```text
Base model parameters
       ❄️ Frozen
```

Only LoRA adapters are trainable:

```text
LoRA A
LoRA B
       ↓
Trainable
```

Example:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)
```

Now:

```python
for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(name)
```

You will see the trainable adapter parameters.

You can use:

```python
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4,
    weight_decay=0.01
)
```

Then:

```text
Frozen base weights
      ↓
Not updated

LoRA weights
      ↓
Updated
      +
Weight decay applied by optimizer configuration
```

---

# 18. Warmup and weight decay in one complete training loop

Here is a simplified but realistic example.

```python
import torch

from torch.optim import AdamW

from transformers import (
    get_cosine_schedule_with_warmup
)


# ------------------------------------
# Optimizer
# ------------------------------------

optimizer = AdamW(
    model.parameters(),

    lr=1e-4,

    weight_decay=0.01
)


# ------------------------------------
# Training configuration
# ------------------------------------

num_epochs = 3

gradient_accumulation_steps = 4

num_training_steps = (
    len(train_dataloader)
    * num_epochs
    // gradient_accumulation_steps
)

num_warmup_steps = int(
    0.05
    * num_training_steps
)


# ------------------------------------
# Scheduler
# ------------------------------------

scheduler = (
    get_cosine_schedule_with_warmup(
        optimizer,

        num_warmup_steps=num_warmup_steps,

        num_training_steps=num_training_steps
    )
)


# ------------------------------------
# Training
# ------------------------------------

optimizer.zero_grad()

for epoch in range(num_epochs):

    model.train()

    for step, batch in enumerate(
        train_dataloader
    ):

        # Forward pass
        outputs = model(**batch)

        # Scale loss for accumulation
        loss = (
            outputs.loss
            / gradient_accumulation_steps
        )

        # Backpropagation
        loss.backward()

        # Perform actual update
        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
        ):

            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update parameters
            optimizer.step()

            # Update LR
            scheduler.step()

            # Clear gradients
            optimizer.zero_grad()
```

The complete flow:

```text
                     BATCH
                       │
                       ▼
                 Forward pass
                       │
                       ▼
                     Loss
                       │
                       ▼
                  Backward
                       │
                       ▼
              Accumulate gradients
                       │
             Every N micro-batches
                       │
                       ▼
                Gradient clipping
                       │
                       ▼
                 optimizer.step()
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Task gradient       Weight decay
              │                 │
              └────────┬────────┘
                       ▼
                Parameter update
                       │
                       ▼
                scheduler.step()
                       │
                       ▼
                Warmup / decay
```

---

# 19. Hugging Face example

For most fine-tuning projects, you don't manually implement the scheduler.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",

    # -----------------------
    # Learning rate
    # -----------------------

    learning_rate=1e-4,

    # -----------------------
    # Warmup
    # -----------------------

    warmup_ratio=0.05,

    # -----------------------
    # LR scheduler
    # -----------------------

    lr_scheduler_type="cosine",

    # -----------------------
    # Weight decay
    # -----------------------

    weight_decay=0.01,

    # -----------------------
    # Batch configuration
    # -----------------------

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    # -----------------------
    # Training duration
    # -----------------------

    num_train_epochs=3,

    # -----------------------
    # Stability
    # -----------------------

    max_grad_norm=1.0,

    # -----------------------
    # Precision
    # -----------------------

    bf16=True,

    # -----------------------
    # Evaluation
    # -----------------------

    eval_strategy="steps",

    eval_steps=100
)
```

This creates a training configuration where:

```text
Initial training
      ↓
Warmup learning rate

      ↓
Target learning rate

      ↓
Cosine decay

      ↓
Smaller learning rate near end
```

At the same time:

```text
Trainable parameters
       ↓
AdamW update
       +
Weight decay regularization
```

---

# 20. How to choose warmup and weight decay?

A practical starting point for experiments might be:

```python
learning_rate = 1e-4

warmup_ratio = 0.03

weight_decay = 0.01
```

Then compare configurations.

Example experiment:

| Experiment |   LR | Warmup | Weight decay |
| ---------- | ---: | -----: | -----------: |
| A          | 5e-5 |     3% |         0.01 |
| B          | 1e-4 |     3% |         0.01 |
| C          | 1e-4 |     5% |         0.01 |
| D          | 1e-4 |     3% |         0.05 |

Evaluate:

```text
Training loss
Validation loss
Task accuracy
Response quality
Hallucination
Instruction following
Overfitting
```

---

# 21. Warmup vs Weight Decay

| Feature                | Warmup                       | Weight Decay                            |
| ---------------------- | ---------------------------- | --------------------------------------- |
| Purpose                | Stabilize early optimization | Regularization                          |
| Controls               | Learning rate                | Parameter magnitude                     |
| Happens when?          | Beginning of training        | Throughout training                     |
| Main problem addressed | Unstable initial updates     | Overfitting / excessively large weights |
| Example                | LR: 0 → 1e-4                 | weight_decay=0.01                       |

The mental model:

```text
                    TRAINING

      Warmup                           Weight Decay
         │                                  │
         ▼                                  ▼
  Controls learning rate             Regularizes weights
         │                                  │
         ▼                                  ▼
  Stable early updates              Reduce overfitting risk
```

---

# 22. Strong interview answer

### What is warmup?

> Warmup is a learning-rate strategy where training starts with a small learning rate and gradually increases it to the target learning rate during the initial optimizer steps. In LLM fine-tuning, this can make early optimization more stable and avoid overly aggressive updates before the optimizer state has settled. After warmup, the learning rate may remain constant or follow a decay schedule such as cosine or linear decay.

### What is weight decay?

> Weight decay is a regularization technique that discourages parameters from growing too large. In modern transformer training, AdamW is commonly used because it decouples weight decay from the adaptive gradient update. Weight decay can help reduce overfitting, although the appropriate value depends on the model, dataset, and fine-tuning method.

### The key difference

> **Warmup controls how quickly the learning rate reaches its target value. Weight decay regularizes model parameters throughout training.**

## Final mental model

```text
Training starts
      │
      ▼
Small learning rate
      │
      │  ← WARMUP
      ▼
Target learning rate
      │
      │
      ▼
LR scheduler gradually changes LR
      │
      ▼
Model finishes training


Throughout training:

Trainable weights
      │
      ├── Task gradients
      │
      └── Weight decay
               │
               ▼
        Regularized updates
```

For a typical LLM fine-tuning experiment, a reasonable **starting configuration** could be:

```python
TrainingArguments(
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0
)
```

Then tune based on validation performance and the actual task rather than assuming these values are universally optimal.
