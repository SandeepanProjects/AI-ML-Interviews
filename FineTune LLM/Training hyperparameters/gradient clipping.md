# 1. What is Gradient Clipping?

**Gradient clipping** is a technique used to prevent gradients from becoming too large during training.

It is mainly used to improve **training stability**.

## The training process

During training:

```text
Input
  ↓
Forward pass
  ↓
Prediction
  ↓
Calculate loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer updates weights
```

A parameter update is conceptually:

[
\theta_{new}
============

## \theta_{old}

\eta \nabla_\theta L
]

Where:

* (\theta) = model parameters
* (\eta) = learning rate
* (\nabla_\theta L) = gradient

The problem occurs when the gradient becomes extremely large.

---

# 2. What is the exploding gradient problem?

Suppose:

```text
Learning rate = 0.001
Gradient = 5
```

The update is approximately:

```text
0.001 × 5 = 0.005
```

That is relatively small.

But suppose:

```text
Learning rate = 0.001
Gradient = 10,000
```

Then:

```text
0.001 × 10,000 = 10
```

The model can suddenly make a huge parameter update.

This may cause:

```text
Loss increases
    ↓
Weights become unstable
    ↓
Even larger gradients
    ↓
NaN / Inf
    ↓
Training failure
```

---

# 3. How gradient clipping solves this

Suppose we configure:

```python
max_grad_norm = 1.0
```

Before updating the model:

```text
Original gradient norm = 10
Max allowed norm = 1
```

Gradient clipping scales the gradients down.

```text
Before clipping:

Gradient norm = 10

        ↓

After clipping:

Gradient norm = 1
```

The direction of the gradient is generally preserved; the magnitude is reduced.

---

# 4. What is gradient norm?

A model has many parameters:

```text
w1
w2
w3
w4
...
wn
```

Each parameter has a gradient:

```text
g1
g2
g3
g4
...
gn
```

The L2 norm is:

[
|g|_2
=====

\sqrt{
g_1^2 + g_2^2 + \dots + g_n^2
}
]

For example:

```text
Gradient vector:

[3, 4]
```

Its norm is:

[
\sqrt{3^2 + 4^2}
================

5
]

---

# 5. Clip by norm

Suppose:

```text
Gradient norm = 10
max_grad_norm = 1
```

The scale factor is approximately:

[
\frac{1}{10}
============

0.1
]

So:

```text
Original gradient:

[6, 8]

Norm = 10
```

After clipping:

```text
[0.6, 0.8]

Norm = 1
```

Notice:

```text
Direction stays similar
Magnitude becomes smaller
```

Conceptually:

```text
Large gradient

          ↗
        /
      /
    /
  ●

        ↓ clip

    ● ───↗

Smaller gradient
```

---

# 6. Gradient clipping formula

For norm clipping:

[
g_{clipped}
===========

g
\times
\min
\left(
1,
\frac{max_norm}{|g|}
\right)
]

This means:

### If gradient norm is smaller

```text
Gradient norm = 0.5
Max norm = 1.0
```

Then:

```text
No clipping
```

### If gradient norm is larger

```text
Gradient norm = 10
Max norm = 1.0
```

Then:

```text
Scale gradients down
```

---

# 7. Code: Gradient clipping from scratch

```python
import torch

gradient = torch.tensor([
    3.0,
    4.0
])

max_grad_norm = 1.0

gradient_norm = torch.norm(
    gradient,
    p=2
)

print("Original norm:", gradient_norm.item())


if gradient_norm > max_grad_norm:

    gradient = (
        gradient
        * max_grad_norm
        / gradient_norm
    )


print("Clipped gradient:", gradient)

print(
    "New norm:",
    torch.norm(gradient).item()
)
```

Output approximately:

```text
Original norm: 5.0

Clipped gradient:
tensor([0.6000, 0.8000])

New norm: 1.0
```

---

# 8. Gradient clipping in PyTorch

PyTorch provides:

```python
torch.nn.utils.clip_grad_norm_()
```

Example:

```python
import torch

model = torch.nn.Linear(
    10,
    1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

loss_fn = torch.nn.MSELoss()
```

Training:

```python
for inputs, targets in dataloader:

    # Clear old gradients
    optimizer.zero_grad()

    # Forward pass
    predictions = model(inputs)

    # Calculate loss
    loss = loss_fn(
        predictions,
        targets
    )

    # Backpropagation
    loss.backward()

    # -------------------------
    # Gradient clipping
    # -------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Update parameters
    optimizer.step()
```

The order is important:

```text
Forward
   ↓
Loss
   ↓
loss.backward()
   ↓
Gradients are calculated
   ↓
Gradient clipping
   ↓
optimizer.step()
```

You clip **after `backward()`** and **before `optimizer.step()`**.

---

# 9. Gradient clipping with gradient accumulation

This is especially important in LLM fine-tuning.

Suppose:

```python
gradient_accumulation_steps = 4
```

Correct approach:

```python
optimizer.zero_grad()

for step, batch in enumerate(train_dataloader):

    outputs = model(**batch)

    loss = (
        outputs.loss
        / gradient_accumulation_steps
    )

    loss.backward()

    # Update after accumulation
    if (
        (step + 1)
        % gradient_accumulation_steps
        == 0
    ):

        # Clip accumulated gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        optimizer.zero_grad()
```

Flow:

```text
Micro-batch 1
    ↓
Gradient

Micro-batch 2
    ↓
Gradient accumulated

Micro-batch 3
    ↓
Gradient accumulated

Micro-batch 4
    ↓
Gradient accumulated
    ↓
CLIP GRADIENT
    ↓
optimizer.step()
```

Usually, clipping after accumulation is preferable because you are clipping the gradient that will actually be used for the optimizer update.

---

# 10. Gradient clipping with mixed precision

With FP16 mixed precision, gradients may be scaled.

Typical PyTorch code:

```python
scaler = torch.amp.GradScaler("cuda")

optimizer.zero_grad()

for batch in train_dataloader:

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        outputs = model(**batch)

        loss = outputs.loss

    # Scale loss
    scaler.scale(loss).backward()

    # Unscale gradients FIRST
    scaler.unscale_(optimizer)

    # Clip actual gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Optimizer update
    scaler.step(optimizer)

    scaler.update()

    optimizer.zero_grad()
```

The important part:

```text
Scaled gradients
      ↓
Unscale
      ↓
Gradient clipping
      ↓
Optimizer step
```

If you clip before unscaling, you may clip the artificially scaled gradients incorrectly.

---

# 11. Common values for gradient clipping

Common starting points include:

```text
0.5
1.0
```

Example:

```python
max_grad_norm = 1.0
```

There is no universal best value.

A practical experiment:

| Experiment | Max grad norm |
| ---------- | ------------: |
| A          |           0.5 |
| B          |           1.0 |
| C          |           2.0 |
| D          |   No clipping |

Monitor:

```text
Training stability
Gradient norms
Loss spikes
NaN / Inf
Validation quality
```

For many transformer fine-tuning setups, `1.0` is a common starting point.

---

# 12. Gradient clipping vs reducing learning rate

These solve different problems.

### Lower learning rate

```text
Reduces the size of parameter updates
throughout training.
```

### Gradient clipping

```text
Acts as a safeguard when gradients
occasionally become abnormally large.
```

A useful mental model:

```text
Learning rate
     ↓
Controls normal update size

Gradient clipping
     ↓
Safety limit for abnormal gradients
```

You can use both.

---

# 13. Gradient clipping in Hugging Face

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",

    learning_rate=1e-4,

    max_grad_norm=1.0,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=3
)
```

The trainer applies gradient clipping according to the training configuration.

---

# Part 2: What is Cosine Learning-Rate Scheduling?

A **learning-rate scheduler** changes the learning rate during training.

Instead of:

```text
LR = 0.0001
LR = 0.0001
LR = 0.0001
LR = 0.0001
```

cosine scheduling gradually changes it.

A typical pattern is:

```text
Warmup
   ↓
Maximum learning rate
   ↓
Gradually decreases
   ↓
Small learning rate
```

---

# 14. Why decrease the learning rate?

At the beginning of training:

```text
Model is far from a good solution
```

You may want:

```text
Larger updates
```

Later:

```text
Model is closer to a good solution
```

You often want:

```text
Smaller updates
```

Conceptually:

```text
Beginning
Large steps

        ↓

Middle
Medium steps

        ↓

End
Small steps
```

---

# 15. What does cosine scheduling mean?

The learning rate follows a cosine-shaped curve.

The standard cosine decay formula is:

[
LR(t)
=====

LR_{min}
+
\frac{1}{2}
(LR_{max} - LR_{min})
\left(
1 +
\cos
\left(
\frac{\pi t}{T}
\right)
\right)
]

Where:

* (t) = current training step
* (T) = total decay steps
* (LR_{max}) = maximum learning rate
* (LR_{min}) = minimum learning rate

At the beginning:

```text
cos(0) = 1
```

So:

```text
LR ≈ maximum
```

At the end:

```text
cos(π) = -1
```

So:

```text
LR ≈ minimum
```

---

# 16. Cosine schedule visualization

```text
Learning Rate

Max LR  ───────●
                \
                 \
                  \
                   \
                    \
                     ●──── Min LR

        Training steps →
```

The actual cosine shape is smoother:

```text
LR
│  ●────────
│           \
│            \
│             \
│               \
│                 ●
└──────────────────────→ Steps
```

The learning rate decreases slowly near the beginning and end, with a smoother curve than simple linear decay.

---

# 17. Cosine schedule code from scratch

```python
import math


def cosine_learning_rate(
    step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 0.0
):
    progress = step / total_steps

    cosine_value = math.cos(
        math.pi * progress
    )

    lr = (
        min_lr
        + 0.5
        * (max_lr - min_lr)
        * (1 + cosine_value)
    )

    return lr
```

Use it:

```python
max_lr = 1e-4
min_lr = 1e-6

total_steps = 10000

for step in [
    0,
    2500,
    5000,
    7500,
    10000
]:

    lr = cosine_learning_rate(
        step=step,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )

    print(
        f"Step {step}: "
        f"LR = {lr:.8f}"
    )
```

Approximate output:

```text
Step 0:      LR = 0.00010000
Step 2500:   LR ≈ 0.00008550
Step 5000:   LR ≈ 0.00005050
Step 7500:   LR ≈ 0.00001550
Step 10000:  LR = 0.00000100
```

---

# 18. Warmup + cosine scheduling

This is a common setup for LLM fine-tuning.

Example:

```text
Target LR = 1e-4
Warmup = 500 steps
Total training = 10,000 steps
```

The schedule:

```text
Steps 0–500:

LR
0
 ↓
1e-5
 ↓
5e-5
 ↓
1e-4


Steps 500–10,000:

LR
1e-4
 ↓
Cosine decay
 ↓
Small LR
```

Visualization:

```text
Learning Rate

        Peak LR
           /\
          /  \
         /    \
        /      \
_______/        \________

 Warmup          Cosine decay
```

---

# 19. Code: cosine schedule with warmup

Using Transformers:

```python
from transformers import (
    get_cosine_schedule_with_warmup
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

num_training_steps = 10_000
num_warmup_steps = 500


scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,

    num_warmup_steps=num_warmup_steps,

    num_training_steps=num_training_steps
)
```

Training:

```python
for batch in train_dataloader:

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Update weights
    optimizer.step()

    # Update learning rate
    scheduler.step()
```

The flow:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradient clipping
   ↓
Optimizer update
   ↓
Cosine scheduler
```

---

# 20. Complete LLM fine-tuning example

Here is how everything fits together.

```python
import torch

from torch.optim import AdamW

from transformers import (
    get_cosine_schedule_with_warmup
)


# ---------------------------------
# Training configuration
# ---------------------------------

num_epochs = 3

learning_rate = 1e-4

weight_decay = 0.01

gradient_accumulation_steps = 4

max_grad_norm = 1.0


# ---------------------------------
# Optimizer
# ---------------------------------

optimizer = AdamW(
    model.parameters(),

    lr=learning_rate,

    weight_decay=weight_decay
)


# ---------------------------------
# Calculate training steps
# ---------------------------------

num_training_steps = (
    len(train_dataloader)
    * num_epochs
    // gradient_accumulation_steps
)

warmup_ratio = 0.05

num_warmup_steps = int(
    num_training_steps
    * warmup_ratio
)


# ---------------------------------
# Cosine scheduler
# ---------------------------------

scheduler = (
    get_cosine_schedule_with_warmup(
        optimizer=optimizer,

        num_warmup_steps=num_warmup_steps,

        num_training_steps=num_training_steps
    )
)


# ---------------------------------
# Training loop
# ---------------------------------

optimizer.zero_grad()

for epoch in range(num_epochs):

    model.train()

    for step, batch in enumerate(
        train_dataloader
    ):

        # -----------------------
        # Forward pass
        # -----------------------

        outputs = model(**batch)

        # -----------------------
        # Loss
        # -----------------------

        loss = outputs.loss

        # -----------------------
        # Gradient accumulation
        # -----------------------

        loss = (
            loss
            / gradient_accumulation_steps
        )

        # -----------------------
        # Backpropagation
        # -----------------------

        loss.backward()


        # -----------------------
        # Update model
        # -----------------------

        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
        ):

            # -------------------
            # Gradient clipping
            # -------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            # -------------------
            # Update weights
            # -------------------

            optimizer.step()

            # -------------------
            # Update LR
            # -------------------

            scheduler.step()

            # -------------------
            # Clear gradients
            # -------------------

            optimizer.zero_grad()
```

The complete pipeline:

```text
                    BATCH
                      │
                      ▼
                Forward Pass
                      │
                      ▼
                    Loss
                      │
                      ▼
                 Backward
                      │
                      ▼
             Accumulate Gradients
                      │
                      ▼
              Enough micro-batches?
                      │
                ┌─────┴─────┐
                │           │
               No          Yes
                │           │
                ▼           ▼
             Continue   Gradient Clipping
                            │
                            ▼
                      optimizer.step()
                            │
                            ▼
                       Weight Decay
                            │
                            ▼
                      scheduler.step()
                            │
                            ▼
                 Warmup / Cosine LR
```

---

# 21. Cosine scheduling vs constant learning rate

## Constant LR

```text
LR

1e-4 ─────────────────────────
```

### Advantage

```text
Simple
```

### Disadvantage

```text
May be too aggressive later in training
```

---

## Linear decay

```text
LR

1e-4 \
      \
       \
        \____
```

The learning rate decreases at a constant rate.

---

## Cosine decay

```text
LR

1e-4  ●
       \
        \
         \
           ●
```

The learning rate follows a smooth cosine curve.

---

# 22. Cosine vs linear scheduling

| Feature                | Linear Decay  | Cosine Decay          |
| ---------------------- | ------------- | --------------------- |
| Shape                  | Straight line | Smooth cosine curve   |
| Early decay            | Constant rate | Usually smoother      |
| Late decay             | Constant rate | Flattens near minimum |
| Common in LLM training | Yes           | Yes                   |
| Needs warmup           | Often useful  | Often useful          |

Neither is always universally better.

I would compare them based on:

```text
Validation loss
Task quality
Training stability
Compute budget
```

---

# 23. Gradient clipping vs cosine scheduling

These solve completely different problems.

| Technique         | Purpose                              |
| ----------------- | ------------------------------------ |
| Gradient clipping | Prevent excessively large gradients  |
| Learning rate     | Controls update size                 |
| Warmup            | Stabilizes early LR behavior         |
| Cosine scheduler  | Gradually changes LR during training |
| Weight decay      | Regularizes trainable parameters     |

Mental model:

```text
                         TRAINING

                             │
                             ▼
                      Calculate gradient
                             │
                             ▼
                    Is gradient too large?
                             │
                             ▼
                     Gradient clipping
                             │
                             ▼
                        Optimizer
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
             Learning Rate          Weight Decay
                  │
                  ▼
             LR Scheduler
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
       Warmup         Cosine Decay
```

---

# 24. Strong interview answer

### What is gradient clipping?

> Gradient clipping limits the magnitude of gradients before the optimizer updates the model. It is used to prevent exploding gradients and stabilize training. In transformer fine-tuning, I commonly use norm-based clipping, such as `max_grad_norm=1.0`. After `loss.backward()`, I calculate the gradient norm and, if it exceeds the threshold, scale the gradients down before calling `optimizer.step()`.

### What is cosine learning-rate scheduling?

> Cosine learning-rate scheduling gradually decreases the learning rate according to a cosine curve. A common setup is linear warmup at the beginning, followed by cosine decay toward a small learning rate. The idea is to allow larger updates earlier in training and smaller, more precise updates later.

### Short interview answer

```text
Gradient clipping
    = protects against abnormal gradients.

Cosine scheduling
    = controls how the learning rate decreases over time.
```

# Practical LLM configuration

A common starting configuration:

```python
TrainingArguments(
    learning_rate=1e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    max_grad_norm=1.0,

    weight_decay=0.01,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=3
)
```

The key idea is that all of these work together:

```text
Learning Rate
      ↓
Controls normal update size

Warmup
      ↓
Stabilizes the beginning

Gradient Clipping
      ↓
Protects against extreme gradients

Cosine Schedule
      ↓
Gradually reduces learning rate

Weight Decay
      ↓
Regularizes parameters
```

For an interview, explaining this **combined training loop** clearly is often more valuable than defining each hyperparameter separately.
