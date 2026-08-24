# FP16 vs BF16 — explained properly with code

You wrote **P16**, but I assume you mean **FP16 vs BF16**.

Both are **16-bit floating-point formats** used to reduce GPU memory and accelerate deep learning/LLM training.

The biggest difference is:

> **FP16 has more precision but a smaller numeric range. BF16 has less precision but a much larger numeric range, similar to FP32.**

---

# 1. Why do we need FP16 or BF16?

Traditional deep-learning training uses FP32.

```text
FP32 = 32 bits = 4 bytes
FP16 = 16 bits = 2 bytes
BF16 = 16 bits = 2 bytes
```

For a model with 7 billion parameters:

```text
FP32 weights ≈ 7B × 4 bytes = 28 GB
FP16 weights ≈ 7B × 2 bytes = 14 GB
BF16 weights ≈ 7B × 2 bytes = 14 GB
```

This is only the memory for model weights. During training, you also need memory for:

```text
GPU Memory
│
├── Model weights
├── Gradients
├── Activations
└── Optimizer states
```

Therefore, using 16-bit precision can significantly reduce memory requirements.

---

# 2. How floating-point numbers are stored

A floating-point number is approximately represented as:

[
value = (-1)^{sign} \times mantissa \times 2^{exponent}
]

The bits are divided into:

```text
Sign       → positive or negative
Exponent   → numeric range
Mantissa   → precision
```

Think of:

```text
Exponent = how BIG or SMALL the number can be
Mantissa = how ACCURATELY we represent the number
```

---

# 3. FP32 structure

FP32 uses:

```text
32 bits

┌──────┬────────────┬───────────────────────┐
│ Sign │ Exponent   │ Mantissa              │
├──────┼────────────┼───────────────────────┤
│ 1    │ 8 bits     │ 23 bits               │
└──────┴────────────┴───────────────────────┘
```

So:

```text
Exponent bits = 8
Mantissa bits = 23
```

FP32 gives:

* Large numeric range
* Good precision
* Higher memory usage

---

# 4. FP16 structure

FP16 uses:

```text
16 bits

┌──────┬────────────┬────────────────┐
│ Sign │ Exponent   │ Mantissa       │
├──────┼────────────┼────────────────┤
│ 1    │ 5 bits     │ 10 bits        │
└──────┴────────────┴────────────────┘
```

So:

```text
Exponent = 5 bits
Mantissa = 10 bits
```

This means:

### Advantage

```text
More precision than BF16
```

### Disadvantage

```text
Much smaller numeric range
```

FP16 can approximately represent:

```text
Smallest normal positive ≈ 6 × 10^-5
Largest finite ≈ 65504
```

This limited range can create problems during training.

---

# 5. BF16 structure

BF16 means **Brain Floating Point 16**.

It uses:

```text
16 bits

┌──────┬────────────┬─────────┐
│ Sign │ Exponent   │ Mantissa│
├──────┼────────────┼─────────┤
│ 1    │ 8 bits     │ 7 bits  │
└──────┴────────────┴─────────┘
```

So:

```text
Exponent = 8 bits
Mantissa = 7 bits
```

Notice something important:

```text
FP32 exponent = 8 bits
BF16 exponent = 8 bits
```

Therefore:

> **BF16 has approximately the same numeric range as FP32.**

But:

```text
FP32 mantissa = 23 bits
BF16 mantissa = 7 bits
```

So BF16 has less numerical precision.

---

# 6. The most important comparison

```text
                     FP32
┌──────┬──────────┬────────────────────────┐
│ Sign │ Exponent │ Mantissa               │
│  1   │    8     │   23                   │
└──────┴──────────┴────────────────────────┘


                     FP16
┌──────┬──────────┬───────────┐
│ Sign │ Exponent │ Mantissa  │
│  1   │    5     │   10      │
└──────┴──────────┴───────────┘


                     BF16
┌──────┬──────────┬───────────┐
│ Sign │ Exponent │ Mantissa  │
│  1   │    8     │    7      │
└──────┴──────────┴───────────┘
```

## Summary

| Format | Total bits | Exponent | Mantissa |
| ------ | ---------: | -------: | -------: |
| FP32   |         32 |        8 |       23 |
| FP16   |         16 |        5 |       10 |
| BF16   |         16 |        8 |        7 |

This gives us the key insight:

```text
FP16:
More precision
Less range

BF16:
Less precision
Much more range
```

---

# 7. What does "range" mean?

Consider these numbers:

```text
0.00000000001
0.001
1
100
100000
1000000000000
```

A floating-point format needs to represent both:

```text
Very small numbers
        ↓
0.00000000001

Very large numbers
        ↓
1000000000000
```

The **exponent bits** determine how large or small the number can be.

Since FP16 has only:

```text
5 exponent bits
```

its range is limited.

BF16 has:

```text
8 exponent bits
```

so its range is much closer to FP32.

---

# 8. Why FP16 can cause training problems

During backpropagation, gradients may be very small.

Example:

```text
Gradient:

0.000000001
```

Suppose the value is too small for FP16's useful representable range.

Conceptually:

```text
Actual gradient
     ↓
0.000000001
     ↓
FP16
     ↓
0
```

This is called **underflow**.

Then:

```text
Gradient = 0
        ↓
Parameter does not update
```

This can hurt training.

---

# 9. FP16 can also overflow

Suppose an activation becomes very large:

```text
100000
```

FP16 maximum is approximately:

```text
65504
```

So:

```text
100000
   ↓
FP16
   ↓
Infinity
```

Then:

```text
Inf
 ↓
NaN
 ↓
Training crashes
```

---

# 10. BF16 helps with range

BF16 has the same number of exponent bits as FP32:

```text
BF16 exponent = 8 bits
FP32 exponent = 8 bits
```

Therefore:

```text
Very small gradient
        ↓
More likely to remain representable

Very large activation
        ↓
Much less likely to overflow compared with FP16
```

This is one reason BF16 is popular for LLM training.

---

# 11. But BF16 has lower precision

Consider:

```text
1.123456789
```

FP32 can represent it more accurately.

FP16:

```text
1.1230
```

BF16 might approximate more aggressively:

```text
1.125
```

The exact stored values depend on rounding and hardware, but conceptually:

```text
FP32
Highest precision

        ↓

FP16
Medium precision

        ↓

BF16
Lower precision
```

However, modern neural networks are often tolerant of this reduced precision.

---

# 12. Code: inspect FP16 vs BF16

```python
import torch

value = torch.tensor(
    [1.123456789],
    dtype=torch.float32
)

fp16_value = value.to(torch.float16)

bf16_value = value.to(torch.bfloat16)

print("FP32:", value)
print("FP16:", fp16_value)
print("BF16:", bf16_value)
```

You may see output similar to:

```text
FP32: tensor([1.1235])

FP16: tensor([1.1230], dtype=torch.float16)

BF16: tensor([1.1250], dtype=torch.bfloat16)
```

BF16 loses more precision around individual values because it has fewer mantissa bits.

---

# 13. Code: compare numerical error

```python
import torch

original = torch.tensor(
    [1.123456789],
    dtype=torch.float32
)

fp16 = original.to(torch.float16).to(torch.float32)

bf16 = original.to(torch.bfloat16).to(torch.float32)


fp16_error = torch.abs(
    original - fp16
)

bf16_error = torch.abs(
    original - bf16
)


print("Original:", original.item())

print("FP16:", fp16.item())
print("BF16:", bf16.item())

print("FP16 error:", fp16_error.item())
print("BF16 error:", bf16_error.item())
```

This demonstrates the important trade-off:

```text
FP16
↓
More precision

BF16
↓
More numeric range
```

---

# 14. Code: demonstrate overflow

```python
import torch

value = torch.tensor(
    [100000.0],
    dtype=torch.float32
)

fp16 = value.to(torch.float16)

bf16 = value.to(torch.bfloat16)


print("FP32:", value)
print("FP16:", fp16)
print("BF16:", bf16)
```

Conceptually:

```text
FP32: 100000

FP16: inf

BF16: approximately 99840
```

BF16 can represent the value, although with reduced precision.

This demonstrates:

```text
FP16
Small range

BF16
Large range
```

---

# 15. FP16 training and loss scaling

Because FP16 has a smaller numeric range, we often use **loss scaling**.

Example:

```python
import torch

scaler = torch.amp.GradScaler("cuda")

for batch in train_dataloader:

    optimizer.zero_grad()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        outputs = model(**batch)

        loss = outputs.loss

    # Scale the loss
    scaler.scale(loss).backward()

    # Unscale gradients
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Optimizer update
    scaler.step(optimizer)

    # Update scaling factor
    scaler.update()
```

The process is:

```text
Loss
 │
 ▼
Scale × 65536
 │
 ▼
Backward in FP16
 │
 ▼
Gradients become larger
 │
 ▼
Less underflow risk
 │
 ▼
Unscale
 │
 ▼
Optimizer update
```

---

# 16. Why must gradients be unscaled before clipping?

Suppose:

```text
Actual gradient = 1
Loss scale = 1000
```

Internally:

```text
Scaled gradient = 1000
```

If:

```python
max_grad_norm = 1.0
```

and you clip before unscaling:

```text
Gradient = 1000
        ↓
Clip to 1
```

Then after unscaling:

```text
1 / 1000 = 0.001
```

That is incorrect.

Therefore:

```text
FP16
   ↓
Scale loss
   ↓
Backward
   ↓
Unscale gradients
   ↓
Clip gradients
   ↓
Optimizer step
```

---

# 17. BF16 training code

BF16 usually has enough exponent range that explicit loss scaling is generally unnecessary.

```python
import torch

model = model.cuda()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


for batch in train_dataloader:

    optimizer.zero_grad()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        outputs = model(**batch)

        loss = outputs.loss


    # Standard backward
    loss.backward()


    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    optimizer.step()
```

Flow:

```text
Forward → BF16 autocast
          ↓
Loss
          ↓
Backward
          ↓
Gradient clipping
          ↓
Optimizer
```

---

# 18. Important clarification: mixed precision does NOT mean everything becomes FP16/BF16

Consider:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    output = model(input)
```

This does **not necessarily mean**:

```text
Every operation = FP16
```

Instead:

```text
Autocast
    │
    ├── Safe operations → FP16/BF16
    │
    └── Numerically sensitive operations → higher precision
```

Examples can include operations where higher precision is beneficial for numerical stability.

This is why it is called:

```text
Mixed precision
```

---

# 19. Complete FP16 LLM training example

```python
import torch
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup
)


# --------------------------------
# Configuration
# --------------------------------

learning_rate = 1e-4

num_epochs = 3

gradient_accumulation_steps = 8

max_grad_norm = 1.0


# --------------------------------
# Optimizer
# --------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)


# --------------------------------
# Scheduler
# --------------------------------

total_steps = (
    len(train_dataloader)
    * num_epochs
    // gradient_accumulation_steps
)

scheduler = (
    get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(
            total_steps * 0.05
        ),
        num_training_steps=total_steps
    )
)


# --------------------------------
# FP16 scaler
# --------------------------------

scaler = torch.amp.GradScaler("cuda")


# --------------------------------
# Training
# --------------------------------

optimizer.zero_grad()

for epoch in range(num_epochs):

    model.train()

    for step, batch in enumerate(
        train_dataloader
    ):

        # ----------------------
        # FP16 forward pass
        # ----------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(**batch)

            loss = (
                outputs.loss
                / gradient_accumulation_steps
            )


        # ----------------------
        # Backpropagation
        # ----------------------

        scaler.scale(loss).backward()


        # ----------------------
        # Gradient accumulation
        # ----------------------

        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
        ):

            # Unscale first
            scaler.unscale_(optimizer)

            # Clip
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            # Update model
            scaler.step(optimizer)

            # Update scale
            scaler.update()

            # Update LR
            scheduler.step()

            # Reset gradients
            optimizer.zero_grad()
```

---

# 20. Complete BF16 LLM training example

```python
import torch
from torch.optim import AdamW


learning_rate = 1e-4

gradient_accumulation_steps = 8

max_grad_norm = 1.0


optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)

optimizer.zero_grad()


for epoch in range(3):

    model.train()

    for step, batch in enumerate(
        train_dataloader
    ):

        # ----------------------
        # BF16 forward pass
        # ----------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            outputs = model(**batch)

            loss = (
                outputs.loss
                / gradient_accumulation_steps
            )


        # ----------------------
        # Backward pass
        # ----------------------

        loss.backward()


        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
        ):

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            # Update model
            optimizer.step()

            # Clear gradients
            optimizer.zero_grad()
```

---

# 21. Using Hugging Face Trainer

## FP16

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    fp16=True,

    bf16=False,

    learning_rate=1e-4,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_grad_norm=1.0,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05
)
```

## BF16

```python
training_args = TrainingArguments(
    output_dir="./output",

    fp16=False,

    bf16=True,

    learning_rate=1e-4,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_grad_norm=1.0,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05
)
```

Typically:

```text
Choose one:

fp16=True

OR

bf16=True
```

Not both.

---

# 22. FP16 vs BF16 vs FP32

| Feature        |    FP32 |             FP16 |                              BF16 |
| -------------- | ------: | ---------------: | --------------------------------: |
| Total bits     |      32 |               16 |                                16 |
| Exponent bits  |       8 |                5 |                                 8 |
| Mantissa bits  |      23 |               10 |                                 7 |
| Memory         | 4 bytes |          2 bytes |                           2 bytes |
| Numeric range  |   Large |          Smaller |                   Similar to FP32 |
| Precision      |    High | Better than BF16 |                   Lower than FP16 |
| Overflow risk  |     Low |           Higher |                             Lower |
| Underflow risk |     Low |           Higher |                             Lower |
| Loss scaling   |      No |  Commonly needed |                Usually not needed |
| LLM training   |     Yes |              Yes | Very common on supported hardware |

---

# 23. Which should you use?

A practical rule:

### Use BF16 when:

```text
Modern GPU/accelerator supports BF16
        ↓
You are training/fine-tuning an LLM
        ↓
You want better numerical range/stability
```

### Use FP16 when:

```text
Hardware supports FP16 but BF16 is unavailable
        ↓
You want reduced memory usage
        ↓
You use proper loss scaling
```

### Use FP32 when:

```text
Maximum numerical precision is required
OR
Debugging numerical instability
OR
Hardware does not efficiently support lower precision
```

---

# 24. FP16 vs BF16 in QLoRA

This is particularly important for your LLM fine-tuning questions.

In QLoRA:

```text
Base Model
    ↓
Stored in 4-bit NF4
    ↓
Frozen

        +

LoRA Adapters
    ↓
Trainable
    ↓
Usually BF16/FP16
```

Conceptually:

```text
        4-bit Base Model
              │
              │ Frozen
              ▼
        ┌──────────────┐
        │  Transformer │
        │   Layers     │
        └──────────────┘
              +
              │
      LoRA adapters
       FP16/BF16
              │
              ▼
         Trainable
```

Example:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

Here:

```text
Base weights storage
        ↓
4-bit NF4

Computation
        ↓
BF16
```

Then you can attach LoRA adapters:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

---

# 25. Strong interview answer

> **FP16 and BF16 are both 16-bit floating-point formats used to reduce memory usage and accelerate deep learning training. FP16 has 5 exponent bits and 10 mantissa bits, so it provides relatively more precision but has a limited numeric range, making overflow and underflow more likely. BF16 has 8 exponent bits, the same as FP32, and 7 mantissa bits, so it has lower precision but a much larger numeric range and is generally more numerically stable for modern LLM training. FP16 commonly uses gradient or loss scaling, while BF16 usually does not require it.**

## The easiest way to remember

```text
                 FP16

        More precision
              +
        Less numeric range
              ↓
        Often needs loss scaling


                 BF16

        Less precision
              +
        Much larger numeric range
              ↓
        Often more stable for LLMs
```

**One-line interview answer:**

> **FP16 prioritizes precision, while BF16 prioritizes numeric range. For modern LLM training, BF16 is often preferred when supported because its FP32-like exponent range reduces overflow and underflow problems.**
