# QLoRA: 4-bit Quantization, NF4, Double Quantization, and Paged Optimizers

These four concepts are closely connected. Let’s build them step by step.

---

# 1. What is 4-bit quantization?

## Basic idea

A neural network normally stores weights using floating-point numbers.

For example:

```text
FP32 → 32 bits per value
FP16 → 16 bits per value
BF16 → 16 bits per value
```

Example weights:

```text
0.12345
-0.78231
1.34567
-0.04521
```

These values require many bits to store accurately.

With **4-bit quantization**, we approximate the weights using only **4 bits**.

```text
FP16 weight
      │
      ▼
Quantization
      │
      ▼
4-bit representation
```

Since 4 bits can represent:

[
2^4 = 16
]

different codes/levels.

So instead of storing a full floating-point value directly, we store a 4-bit code plus quantization metadata such as scales.

---

## Simple analogy

Imagine a temperature:

```text
23.4567°C
```

With high precision:

```text
23.4567
```

With lower precision:

```text
23.5
```

You lose some precision but save storage.

Quantization does the same thing with neural-network weights:

```text
FP16/BF16 weights
        ↓
Approximate
        ↓
4-bit weights
```

---

# 2. Why does 4-bit quantization save memory?

Suppose we have:

```text
1 billion parameters
```

### FP32

Each parameter:

```text
4 bytes
```

Memory:

[
1B \times 4
===========

4GB
]

### FP16

Each parameter:

```text
2 bytes
```

Memory:

[
1B \times 2
===========

2GB
]

### 4-bit

Each parameter:

[
4/8 = 0.5
]

bytes.

Raw storage:

[
1B \times 0.5
=============

0.5GB
]

So:

| Precision | Bits | Raw bytes/parameter | Approximate raw memory for 1B parameters |
| --------- | ---: | ------------------: | ---------------------------------------: |
| FP32      |   32 |                   4 |                                     4 GB |
| FP16      |   16 |                   2 |                                     2 GB |
| 4-bit     |    4 |                 0.5 |                                   0.5 GB |

Actual memory is higher because quantization also needs metadata and runtime buffers.

---

# 3. How does 4-bit quantization work?

Suppose the original weights are:

```text
[-1.2, -0.8, -0.3, 0.1, 0.4, 0.9, 1.3]
```

A simple quantizer might map values to 16 levels.

For example:

```text
Original      Quantized code

-1.2     →      0
-0.8     →      2
-0.3     →      5
 0.1     →      8
 0.4     →      10
 0.9     →      13
 1.3     →      15
```

Instead of storing:

```text
0.423876
```

you might store:

```text
1010
```

which is a 4-bit code.

To reconstruct approximately:

```text
4-bit code
    +
scale/metadata
    ↓
Approximate floating-point value
```

---

# 4. Simple code: uniform 4-bit quantization

Here is a simplified educational implementation.

```python
import torch


def quantize_4bit(weights):

    # Find minimum and maximum
    min_val = weights.min()
    max_val = weights.max()

    # 4-bit = 16 levels
    levels = 16

    # Calculate quantization scale
    scale = (
        max_val - min_val
    ) / (levels - 1)

    # Convert weights to integers 0-15
    quantized = torch.round(
        (weights - min_val) / scale
    )

    # Clamp values
    quantized = torch.clamp(
        quantized,
        0,
        15
    ).to(torch.uint8)

    return quantized, scale, min_val
```

Dequantization:

```python
def dequantize_4bit(
    quantized,
    scale,
    min_val
):

    weights = (
        quantized.float()
        * scale
        + min_val
    )

    return weights
```

Example:

```python
weights = torch.tensor([
    -1.2,
    -0.8,
    -0.3,
    0.1,
    0.4,
    0.9,
    1.3
])

quantized, scale, min_val = quantize_4bit(
    weights
)

reconstructed = dequantize_4bit(
    quantized,
    scale,
    min_val
)

print("Original:")
print(weights)

print("\n4-bit codes:")
print(quantized)

print("\nReconstructed:")
print(reconstructed)
```

You will see:

```text
Original weights
        ↓
Quantized 4-bit codes
        ↓
Approximate reconstructed weights
```

The reconstructed values are not exactly identical because quantization is **lossy compression**.

---

# 5. What is NF4?

**NF4 stands for NormalFloat 4-bit.**

It is a special 4-bit data type designed for neural-network weights that are approximately **normally distributed**.

This is important because LLM weights often look roughly like this:

```text
              ███
           ███████
         ███████████
       ███████████████
───────0────────────────
```

Most values are near:

```text
0
```

while relatively few values are very large.

---

## The problem with uniform quantization

Suppose you have 16 levels:

```text
-1.0
-0.87
-0.74
-0.61
...
0
...
0.74
0.87
1.0
```

This distributes levels approximately uniformly.

But neural-network weights are not uniformly distributed.

There are many values near zero:

```text
-0.05
0.01
0.02
-0.03
0.08
```

and fewer values near extremes:

```text
-0.95
0.92
```

So uniform spacing can waste representation capacity.

---

# 6. NF4 uses levels better

NF4 places its representable values to better match a normal distribution.

Conceptually:

```text
Uniform quantization:

|----|----|----|----|----|----|


NF4-like distribution:

|--|-|-|---|---|---|---|---|---|-|-|--|

       More precision near zero
```

So NF4 generally allocates more useful resolution where neural-network weights are concentrated.

The important idea is:

> **NF4 is a 4-bit quantization scheme designed around the approximate normal distribution of neural-network weights.**

---

# 7. NF4 in QLoRA

In QLoRA:

```text
Original Model Weights
          │
          ▼
Block-wise scaling / normalization
          │
          ▼
NF4 Quantization
          │
          ▼
4-bit Stored Weights
```

During computation:

```text
NF4 weights
     │
     ▼
Dequantization for computation
     │
     ▼
Matrix multiplication
```

The base model remains stored efficiently while computations use suitable higher-precision representations where required.

---

# 8. NF4 code in QLoRA

Using `bitsandbytes` through Transformers:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
```

Configuration:

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

The key line is:

```python
bnb_4bit_quant_type="nf4"
```

Then:

```python
model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

Now the model's quantized weights use NF4.

---

# 9. What is Double Quantization?

This is one of the more confusing QLoRA concepts.

Let's first understand regular quantization.

---

## Normal quantization requires metadata

Suppose you divide weights into blocks:

```text
Block 1:
[w1, w2, w3, ...]

Block 2:
[w1, w2, w3, ...]

Block 3:
[w1, w2, w3, ...]
```

Each block is quantized.

To reconstruct the weights, you need information such as:

```text
scale
```

So:

```text
4-bit weights
+
scale values
```

The weights are compressed, but the scale values still consume memory.

For a huge model:

```text
Billions of weights
+
many quantization constants
```

can create meaningful overhead.

---

## Double Quantization quantizes the quantization constants

Normally:

```text
Original weights
       │
       ▼
Quantization
       │
       ├── 4-bit weight codes
       │
       └── FP16/FP32 scale constants
```

Double quantization:

```text
Original weights
       │
       ▼
First quantization
       │
       ├── 4-bit weights
       │
       ▼
Quantize scale constants
       │
       ▼
Compressed scale information
```

Therefore:

[
\boxed{
\text{Double Quantization}
==========================

\text{Quantizing the quantization metadata}
}
]

This further reduces memory usage.

---

# 10. Conceptual code for double quantization

Educational example:

```python
import torch
```

First, imagine we have scale values:

```python
scales = torch.tensor([
    0.002,
    0.005,
    0.010,
    0.020,
    0.050
])
```

These scales normally require floating-point storage.

Now quantize them:

```python
def quantize_values(values, bits=8):

    levels = 2 ** bits

    min_val = values.min()
    max_val = values.max()

    scale = (
        max_val - min_val
    ) / (levels - 1)

    quantized = torch.round(
        (values - min_val) / scale
    )

    return (
        quantized.to(torch.uint8),
        scale,
        min_val
    )
```

Use it:

```python
quantized_scales, scale_scale, scale_min = (
    quantize_values(
        scales,
        bits=8
    )
)

print(quantized_scales)
```

Conceptually:

```text
Original scales

0.002
0.005
0.010
0.020
0.050

        ↓

Quantized scales

integer codes
+
small reconstruction metadata
```

Real QLoRA implementations are more sophisticated than this educational example.

---

# 11. QLoRA double quantization configuration

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

The important line:

```python
bnb_4bit_use_double_quant=True
```

This enables nested/double quantization in the bitsandbytes implementation.

---

# 12. What are Paged Optimizers?

Paged optimizers were introduced in the QLoRA work to help handle **GPU memory spikes during training**.

To understand this, let's first look at optimizer memory.

---

## Adam optimizer uses extra memory

Suppose you train parameters:

```text
Trainable Weights
```

Adam typically maintains optimizer state such as:

```text
First moment (m)
Second moment (v)
```

Conceptually:

```text
Parameters
    +
Gradients
    +
Adam m
    +
Adam v
```

This can consume significant memory.

---

# 13. What is the memory spike problem?

During training, memory usage is not always constant.

```text
GPU Memory

██████████████

Forward pass

██████████████████

Backward pass

████████████████████████

Optimizer step

██████████████████████████
                         ↑
                     Memory spike
```

If GPU memory exceeds the available capacity:

```text
CUDA Out Of Memory
```

Even if normal memory usage is acceptable.

---

# 14. Paged optimizer concept

A paged optimizer uses a paging/unified-memory approach to manage optimizer-state memory.

Conceptually:

```text
GPU Memory
─────────────────────

Active optimizer state
        │
        ▼

GPU memory pressure?
        │
       Yes
        │
        ▼

Some optimizer state can be managed through
paged memory / unified memory mechanisms
```

Visual model:

```text
                    GPU Memory
               ┌─────────────────┐
               │                 │
               │ Model           │
               │ Activations     │
               │ LoRA parameters │
               │                 │
               └────────┬────────┘
                        │
                  Memory pressure
                        │
                        ▼
                  Paged state
                        │
                        ▼
                CPU / Unified Memory
```

The goal is to reduce the chance that temporary memory spikes immediately cause OOM.

---

# 15. Important clarification about paged optimizers

Paged optimizers do **not** mean:

```text
GPU can train infinitely large models
```

They also do not mean:

```text
Everything is always faster
```

Moving or paging memory can introduce overhead.

A better explanation is:

> **Paged optimizers use memory paging/unified memory techniques to manage optimizer-state memory under GPU memory pressure, helping reduce out-of-memory failures caused by memory spikes.**

---

# 16. Example configuration

In a Trainer-style setup, conceptually:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(

    output_dir="./output",

    optim="paged_adamw_32bit",

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8
)
```

Another possible configuration is:

```python
optim="paged_adamw_8bit"
```

The exact options available depend on your installed versions of:

* `transformers`
* `bitsandbytes`

---

# 17. Full QLoRA architecture

Now combine everything.

```text
                    Base LLM

             FP16/BF16 original weights
                       │
                       ▼
                NF4 Quantization
                       │
                       ▼
                4-bit Base Model
                       │
                       ├───────────────┐
                       │               │
                       │          LoRA Adapters
                       │          FP16/BF16
                       │          Trainable ✓
                       │               │
                       └───────┬───────┘
                               ▼
                             Output
                               │
                               ▼
                              Loss
                               │
                               ▼
                        Backpropagation
                               │
                               ▼
                     Update LoRA Only ✓
```

And memory optimization:

```text
4-bit Quantization
        │
        ▼
Smaller Base Model Memory


NF4
        │
        ▼
Better 4-bit representation
for normally distributed weights


Double Quantization
        │
        ▼
Smaller quantization metadata


Paged Optimizers
        │
        ▼
Better handling of memory spikes
```

---

# 18. Complete QLoRA configuration

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model
)
```

## Configure quantization

```python
bnb_config = BitsAndBytesConfig(

    # Load base model in 4-bit
    load_in_4bit=True,

    # Use NF4 quantization
    bnb_4bit_quant_type="nf4",

    # Quantize quantization constants
    bnb_4bit_use_double_quant=True,

    # Compute in BF16
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

## Load model

```python
model = AutoModelForCausalLM.from_pretrained(

    "your-model",

    quantization_config=bnb_config,

    device_map="auto"
)
```

---

## Prepare for training

```python
model = prepare_model_for_kbit_training(
    model
)
```

---

## Configure LoRA

```python
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

Check:

```python
model.print_trainable_parameters()
```

Expected conceptually:

```text
Base Model
4-bit
Frozen ❄️

LoRA Adapters
FP16/BF16
Trainable ✓
```

---

# 19. Difference between the four concepts

| Concept             | Main Purpose                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| 4-bit Quantization  | Reduce model-weight memory                                                |
| NF4                 | Improve 4-bit quantization for approximately normally distributed weights |
| Double Quantization | Reduce quantization metadata overhead                                     |
| Paged Optimizers    | Manage optimizer-state memory and spikes                                  |

---

# 20. Best interview answer

> **4-bit quantization compresses model weights into 4-bit representations, reducing raw weight storage substantially compared with FP16. NF4 is a specialized 4-bit data type designed for approximately normally distributed neural-network weights, improving quantization efficiency for that distribution. Double quantization further reduces memory by quantizing the quantization constants themselves. Paged optimizers use paging or unified-memory techniques to manage optimizer-state memory under GPU memory pressure and help avoid out-of-memory failures from temporary memory spikes. Together, these techniques make QLoRA much more memory-efficient for fine-tuning large language models.**

## Easy memory trick

```text
4-bit
↓
Compress weights

NF4
↓
Better 4-bit representation
for normal-like weight distributions

Double Quantization
↓
Compress quantization metadata

Paged Optimizer
↓
Manage optimizer memory pressure
```

[
\boxed{
QLoRA =
4\text{-bit}
+
NF4
+
Double\ Quantization
+
LoRA
+
Paged\ Optimizers
}
]

A small correction to the last formula: **QLoRA fundamentally means a quantized frozen base model plus LoRA adapters**. NF4, double quantization, and paged optimizers are important techniques from the original QLoRA approach, but they are not mathematically required in every implementation called “QLoRA.”
