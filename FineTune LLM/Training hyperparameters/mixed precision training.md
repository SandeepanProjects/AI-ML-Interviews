# 1. What is Linear Learning-Rate Decay?

**Linear learning-rate decay** is a learning-rate scheduling strategy where the learning rate gradually decreases in a straight-line manner during training.

The basic idea is:

```text
Beginning of training          End of training

High LR                              Low LR

1e-4 ────\ 
          \
           \
            \
             \____ 0
```

Instead of keeping the learning rate constant:

```text
Step 1     → 0.0001
Step 1000  → 0.0001
Step 5000  → 0.0001
Step 10000 → 0.0001
```

we gradually reduce it:

```text
Step 1     → 0.0001
Step 2500  → 0.000075
Step 5000  → 0.000050
Step 7500  → 0.000025
Step 10000 → 0.000000
```

---

# 2. Why decay the learning rate?

Think of training as finding the bottom of a valley.

At the beginning:

```text
Model is far from optimum
        ↓
Need relatively larger steps
```

Near the end:

```text
Model is closer to optimum
        ↓
Need smaller, more precise steps
```

Example:

```text
Beginning:

      🟢
       \
        \
         \____

Large steps help move quickly.


Later:

             🟢
              \
               \____

Smaller steps help avoid overshooting.
```

So:

```text
Early training
      ↓
Higher learning rate
      ↓
Faster learning

Late training
      ↓
Lower learning rate
      ↓
More stable convergence
```

---

# 3. Linear decay formula

Suppose:

```python
initial_lr = 1e-4
total_steps = 10_000
```

The linear decay formula is approximately:

[
LR(t) = LR_{initial} \times \left(1 - \frac{t}{T}\right)
]

Where:

* (t) = current training step
* (T) = total training steps
* (LR_{initial}) = starting learning rate

Example:

At step 0:

[
LR = 10^{-4} \times (1 - 0)
]

```text
LR = 0.0001
```

At step 5,000:

[
LR = 10^{-4} \times \left(1-\frac{5000}{10000}\right)
]

```text
LR = 0.00005
```

At step 10,000:

```text
LR = 0
```

---

# 4. Code: implement linear decay from scratch

```python
def get_linear_lr(
    step: int,
    total_steps: int,
    initial_lr: float,
    min_lr: float = 0.0
) -> float:

    progress = min(
        step / total_steps,
        1.0
    )

    lr = (
        initial_lr
        + (min_lr - initial_lr)
        * progress
    )

    return lr


initial_lr = 1e-4
min_lr = 1e-6
total_steps = 10_000


for step in [
    0,
    2500,
    5000,
    7500,
    10000
]:

    lr = get_linear_lr(
        step=step,
        total_steps=total_steps,
        initial_lr=initial_lr,
        min_lr=min_lr
    )

    print(
        f"Step {step}: "
        f"LR = {lr:.8f}"
    )
```

Approximate output:

```text
Step 0:     LR = 0.00010000
Step 2500:  LR = 0.00007525
Step 5000:  LR = 0.00005050
Step 7500:  LR = 0.00002575
Step 10000: LR = 0.00000100
```

---

# 5. Linear decay + warmup

For LLM training, we often don't immediately start with the maximum learning rate.

Instead:

```text
Phase 1
Warmup

LR:
0
↓
↓
↓
Maximum LR
```

Then:

```text
Phase 2
Linear decay

Maximum LR
↓
↓
↓
Minimum LR
```

Visualization:

```text
Learning Rate

           Maximum LR
               /\
              /  \
             /    \
            /      \
           /        \
__________/          \________

   Warmup             Linear decay
```

## Formula

### Warmup phase

[
LR(t) =
LR_{max}
\times
\frac{t}{warmup_steps}
]

### Decay phase

[
LR(t)
=====

LR_{max}
\times
\left(
1 -
\frac{
t-warmup_steps
}{
total_steps-warmup_steps
}
\right)
]

---

# 6. Code: linear warmup + linear decay

```python
def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float
) -> float:

    # ---------------------------
    # Warmup phase
    # ---------------------------

    if step < warmup_steps:

        return (
            max_lr
            * step
            / warmup_steps
        )

    # ---------------------------
    # Decay phase
    # ---------------------------

    progress = (
        (step - warmup_steps)
        / (total_steps - warmup_steps)
    )

    return max(
        0.0,
        max_lr * (1 - progress)
    )


max_lr = 1e-4
warmup_steps = 1000
total_steps = 10_000


for step in [
    0,
    500,
    1000,
    2500,
    5000,
    7500,
    10000
]:

    lr = get_lr(
        step,
        warmup_steps,
        total_steps,
        max_lr
    )

    print(
        f"Step {step}: "
        f"{lr:.8f}"
    )
```

Output approximately:

```text
Step 0:      0.00000000
Step 500:    0.00005000
Step 1000:   0.00010000
Step 2500:   0.00008333
Step 5000:   0.00005556
Step 7500:   0.00002778
Step 10000:  0.00000000
```

This is the complete idea:

```text
                    LR

                    1e-4
                     ▲
                    / \
                   /   \
                  /     \
                 /       \
                /         \
───────────────             ────────► steps
        Warmup       Linear decay
```

---

# 7. Linear scheduling using Hugging Face

You normally don't write the scheduler manually.

```python
import torch

from torch.optim import AdamW

from transformers import (
    get_linear_schedule_with_warmup
)
```

Create optimizer:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

Create scheduler:

```python
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

    # Forward pass
    outputs = model(**batch)

    loss = outputs.loss

    # Backpropagation
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Update model weights
    optimizer.step()

    # Update learning rate
    scheduler.step()
```

The order is:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradient clipping
   ↓
optimizer.step()
   ↓
scheduler.step()
```

---

# 8. Linear decay using `TrainingArguments`

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="linear",

    weight_decay=0.01,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_grad_norm=1.0
)
```

The Hugging Face trainer handles:

```text
Optimizer
      +
Warmup
      +
Linear LR decay
      +
Gradient accumulation
      +
Gradient clipping
```

---

# 9. Linear decay vs cosine decay

## Linear

```text
LR

●
 \
  \
   \
    \
     ●
```

The learning rate decreases at a constant rate.

## Cosine

```text
LR

●
 \
  \
   )
    )
     ●
```

Cosine decay follows a smooth curve and usually changes more gradually near the beginning and end.

| Feature                | Linear        | Cosine                      |
| ---------------------- | ------------- | --------------------------- |
| Shape                  | Straight line | Curved                      |
| Easy to understand     | Yes           | Yes                         |
| Common for LLMs        | Yes           | Yes                         |
| Often used with warmup | Yes           | Yes                         |
| Requires tuning        | Less complex  | May require experimentation |

Neither is universally best.

---

# Part 2: What is Mixed Precision Training?

**Mixed precision training** means using multiple numerical precisions during training instead of performing every operation in FP32.

The common precisions are:

```text
FP32  → 32-bit floating point
FP16  → 16-bit floating point
BF16  → 16-bit floating point format
```

Traditional training:

```text
Everything

FP32
 │
 ▼
Model
 │
 ▼
FP32 calculations
 │
 ▼
FP32 gradients
```

Mixed precision:

```text
Some operations → FP16/BF16
Other operations → FP32
```

This is called **mixed precision** because different operations use different precisions.

---

# 10. Why do we need mixed precision?

LLMs require a huge amount of memory.

GPU memory is used for:

```text
GPU Memory
│
├── Model weights
│
├── Gradients
│
├── Optimizer states
│
└── Activations
```

FP32 requires:

```text
4 bytes per value
```

FP16/BF16 requires:

```text
2 bytes per value
```

Example:

```text
1 billion parameters

FP32
≈ 4 GB for weights

FP16/BF16
≈ 2 GB for weights
```

This is only a simplified estimate for model weights; actual training memory is much higher because gradients, activations, and optimizer states also consume memory.

Mixed precision can provide:

```text
Less memory
      +
Potentially faster computation
      +
Larger batch sizes
      +
Ability to train larger models
```

---

# 11. FP32 vs FP16 vs BF16

## FP32

```text
32 bits

High precision
Large memory
Usually stable
```

## FP16

```text
16 bits

Less memory
Fast on supported hardware
Smaller numeric range than FP32
Can be more susceptible to overflow/underflow
```

## BF16

```text
16 bits

Less memory
Similar exponent range to FP32
Less precision in the mantissa
Often convenient for modern LLM training hardware
```

A simplified comparison:

| Type | Bits | Memory/value | Numeric behavior                                |
| ---- | ---: | -----------: | ----------------------------------------------- |
| FP32 |   32 |      4 bytes | High precision/range                            |
| FP16 |   16 |      2 bytes | Reduced range and precision                     |
| BF16 |   16 |      2 bytes | Wider range than FP16, lower mantissa precision |

---

# 12. Why not use FP16 for everything?

Suppose a very small gradient is:

```text
0.00000001
```

With limited precision/range, very small values can become difficult to represent accurately.

Conceptually:

```text
True gradient
   ↓
Very small value
   ↓
Low precision representation
   ↓
May become zero
```

This is called **underflow**.

Similarly, large values can cause:

```text
Overflow
   ↓
Inf
   ↓
NaN
```

Therefore, training often keeps some operations or states in higher precision.

---

# 13. What is loss scaling?

Loss scaling is especially associated with FP16 training.

Suppose the loss is:

```text
loss = 0.00001
```

Before backpropagation:

```text
Scaled loss =
loss × scale_factor
```

Example:

```text
0.00001 × 1024
=
0.01024
```

Now gradients are larger and less likely to underflow.

After gradients are calculated:

```text
Unscale gradients
```

Conceptually:

```text
Original loss
     │
     ▼
Scale loss
     │
     ▼
Backward
     │
     ▼
Large gradients
     │
     ▼
Unscale gradients
     │
     ▼
Clip gradients
     │
     ▼
Optimizer step
```

---

# 14. Mixed precision training using PyTorch

Modern PyTorch uses `autocast`.

```python
import torch

model = model.cuda()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

scaler = torch.amp.GradScaler("cuda")
```

Training:

```python
for batch in train_dataloader:

    optimizer.zero_grad()

    # Mixed precision region
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        outputs = model(**batch)

        loss = outputs.loss

    # Scale loss
    scaler.scale(loss).backward()

    # Unscale gradients
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Optimizer step
    scaler.step(optimizer)

    # Update scale factor
    scaler.update()
```

---

# 15. What happens internally?

```text
                   INPUT
                     │
                     ▼
              autocast context
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     Safe operations       Sensitive operations
          │                     │
          ▼                     ▼
       FP16/BF16                FP32
          │                     │
          └──────────┬──────────┘
                     ▼
                   LOSS
                     │
                     ▼
              Loss Scaling
                     │
                     ▼
                  BACKWARD
                     │
                     ▼
             Unscale gradients
                     │
                     ▼
             Gradient clipping
                     │
                     ▼
               Optimizer step
```

The framework determines many casting decisions automatically.

---

# 16. BF16 training

For modern LLM training, BF16 is frequently used when the hardware supports it.

Example:

```python
import torch

for batch in train_dataloader:

    optimizer.zero_grad()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        outputs = model(**batch)

        loss = outputs.loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()
```

A major difference is that BF16 often does not require the same loss-scaling approach commonly used with FP16.

---

# 17. FP16 mixed precision with gradient accumulation

A realistic LLM loop:

```python
import torch

gradient_accumulation_steps = 4

optimizer.zero_grad()

scaler = torch.amp.GradScaler("cuda")

for step, batch in enumerate(
    train_dataloader
):

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        outputs = model(**batch)

        loss = (
            outputs.loss
            / gradient_accumulation_steps
        )

    # Scaled backward pass
    scaler.scale(loss).backward()


    # Update every N micro-batches
    if (
        (step + 1)
        % gradient_accumulation_steps
        == 0
    ):

        # Unscale gradients first
        scaler.unscale_(optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update weights
        scaler.step(optimizer)

        # Update scaling factor
        scaler.update()

        # Clear gradients
        optimizer.zero_grad()
```

This combines:

```text
Mixed precision
       +
Gradient accumulation
       +
Gradient clipping
       +
AdamW
```

---

# 18. Complete LLM fine-tuning example

Here is how the concepts fit together.

```python
import torch

from torch.optim import AdamW

from transformers import (
    get_linear_schedule_with_warmup
)


# ---------------------------------
# Configuration
# ---------------------------------

learning_rate = 1e-4

weight_decay = 0.01

num_epochs = 3

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
# Training steps
# ---------------------------------

num_training_steps = (
    len(train_dataloader)
    * num_epochs
    // gradient_accumulation_steps
)

num_warmup_steps = int(
    num_training_steps * 0.05
)


# ---------------------------------
# Scheduler
# ---------------------------------

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# ---------------------------------
# Mixed precision
# ---------------------------------

scaler = torch.amp.GradScaler("cuda")


# ---------------------------------
# Training loop
# ---------------------------------

optimizer.zero_grad()

for epoch in range(num_epochs):

    model.train()

    for step, batch in enumerate(
        train_dataloader
    ):

        # -------------------------
        # Mixed precision forward
        # -------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(**batch)

            loss = (
                outputs.loss
                / gradient_accumulation_steps
            )


        # -------------------------
        # Backward pass
        # -------------------------

        scaler.scale(loss).backward()


        # -------------------------
        # Update after accumulation
        # -------------------------

        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
        ):

            # ---------------------
            # Unscale gradients
            # ---------------------

            scaler.unscale_(optimizer)


            # ---------------------
            # Gradient clipping
            # ---------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )


            # ---------------------
            # Update model
            # ---------------------

            scaler.step(optimizer)


            # ---------------------
            # Update loss scale
            # ---------------------

            scaler.update()


            # ---------------------
            # Update learning rate
            # ---------------------

            scheduler.step()


            # ---------------------
            # Clear gradients
            # ---------------------

            optimizer.zero_grad()
```

---

# 19. Using Hugging Face `Trainer`

You usually don't manually write all of this.

For FP16:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",

    # Learning rate
    learning_rate=1e-4,

    # Warmup + linear decay
    warmup_ratio=0.05,
    lr_scheduler_type="linear",

    # Regularization
    weight_decay=0.01,

    # Gradient clipping
    max_grad_norm=1.0,

    # Batch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    # Precision
    fp16=True,

    # Epochs
    num_train_epochs=3
)
```

For BF16:

```python
training_args = TrainingArguments(
    output_dir="./model",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="linear",

    weight_decay=0.01,

    max_grad_norm=1.0,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    bf16=True,

    num_train_epochs=3
)
```

Don't enable both:

```python
fp16=True
bf16=True
```

Use the precision mode supported by your hardware and framework.

---

# 20. Linear decay vs cosine decay

| Feature             | Linear Decay              | Cosine Decay               |
| ------------------- | ------------------------- | -------------------------- |
| Learning-rate shape | Straight line             | Cosine curve               |
| Beginning           | Maximum after warmup      | Maximum after warmup       |
| Middle              | Constant rate of decrease | Smooth nonlinear decrease  |
| End                 | Reaches low/end LR        | Smoothly approaches low LR |
| Complexity          | Simple                    | Slightly more complex      |

Example:

```text
Linear:

LR
│\
│ \
│  \
│   \
│    \
└─────→ steps


Cosine:

LR
│●
│ \
│  \
│   )
│    )
│     ●
└──────→ steps
```

---

# 21. Mixed precision vs quantization

These are often confused.

## Mixed precision training

```text
Purpose:
Reduce training memory and potentially speed up training.

Example:
FP16 + FP32
or
BF16 + FP32
```

## Quantization

```text
Purpose:
Reduce model memory and inference/training cost.

Example:
FP16
INT8
4-bit NF4
```

For example, QLoRA uses:

```text
Quantized base model
       ↓
4-bit NF4

LoRA adapters
       ↓
Higher precision training
```

Mixed precision and quantization can both be present in a training setup, but they solve different problems.

---

# 22. Strong interview answer

### What is linear learning-rate decay?

> Linear learning-rate decay gradually decreases the learning rate from its maximum value to a smaller value over the training steps at a constant rate. In LLM fine-tuning, it is commonly combined with an initial warmup phase. Warmup allows the optimizer to gradually reach the target learning rate, and linear decay then reduces the learning rate so later training updates become smaller and more stable.

### What is mixed precision training?

> Mixed precision training uses multiple numerical precisions, typically FP16 or BF16 for many tensor operations and FP32 where higher numerical precision is needed. This reduces GPU memory consumption and can improve training speed. With FP16, gradient scaling is commonly used to reduce underflow risk; gradients are unscaled before clipping and optimizer updates. BF16 has a wider exponent range and often avoids the same loss-scaling requirement.

## Final mental model

```text
                  LLM TRAINING

                        │
                        ▼
                  Forward Pass
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       Mixed Precision        Loss Calculation
          FP16/BF16                  │
              │                      ▼
              └──────────────► Backward
                                      │
                                      ▼
                           Gradient Accumulation
                                      │
                                      ▼
                              Gradient Clipping
                                      │
                                      ▼
                                  AdamW Update
                                      │
                                      ▼
                              Learning Rate
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                     Warmup                    Linear Decay
```

The practical interview takeaway is:

> **Linear learning-rate decay controls how the learning rate decreases over training, while mixed precision controls the numerical precision used for computations to reduce memory usage and improve training efficiency.**
