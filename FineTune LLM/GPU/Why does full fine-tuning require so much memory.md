# Why does full fine-tuning require so much memory?

The short answer:

> During full fine-tuning, the GPU must store not only the model weights, but also gradients, optimizer states, activations, and temporary tensors.

For a 7B model:

```text
Model parameters
      +
Gradients
      +
Optimizer states
      +
Activations
      +
Temporary CUDA buffers
      =
Large GPU memory requirement
```

Let's understand this properly.

---

# 1. What happens during full fine-tuning?

Suppose we have a model:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
 ...
  ↓
Layer N
  ↓
Prediction
  ↓
Loss
```

Training has three major steps:

```text
1. Forward pass
2. Backward pass
3. Optimizer update
```

Each step requires memory.

---

# 2. Model weights require memory

A 7B model has approximately:

```text
7,000,000,000 parameters
```

Each parameter is a number.

For BF16 or FP16:

```text
Each parameter = 2 bytes
```

Therefore:

```text
7B × 2 bytes
≈ 14 GB
```

Python calculation:

```python
parameters = 7_000_000_000

bytes_per_parameter = 2

weight_memory = (
    parameters
    * bytes_per_parameter
)

weight_memory_gb = (
    weight_memory
    / (1024 ** 3)
)

print(
    f"Model weights: "
    f"{weight_memory_gb:.2f} GB"
)
```

Approximate output:

```text
Model weights: 13.04 GB
```

The difference between 13.04 GiB and 14 GB comes from decimal vs binary units.

But during full fine-tuning, storing the weights is only the beginning.

---

# 3. Gradients also require memory

During backpropagation:

```text
Loss
  ↓
Backward pass
  ↓
Calculate gradient for every parameter
```

For every weight:

```text
W
```

we need:

```text
∂Loss
─────
  ∂W
```

Example:

```python
import torch

weight = torch.tensor(
    2.0,
    requires_grad=True
)

loss = (
    weight * weight
)

loss.backward()

print(weight.grad)
```

Output:

```text
tensor(4.)
```

In a neural network:

```text
7B parameters
```

means potentially:

```text
7B gradients
```

So approximately:

```text
Weights   ≈ 14 GB
Gradients ≈ 14 GB
```

Total already:

```text
28 GB
```

---

# 4. Optimizer states require even more memory

This is one of the biggest reasons.

Most LLMs use:

```text
Adam
or
AdamW
```

Adam stores two extra values for every trainable parameter:

```text
m = first moment
v = second moment
```

Conceptually:

```text
Parameter W
     │
     ├── Gradient
     │
     ├── m
     │
     └── v
```

The update is approximately:

```text
m = β₁m + (1 - β₁)g

v = β₂v + (1 - β₂)g²

W = W - learning_rate × m / √v
```

Therefore, for every parameter:

```text
Weight
Gradient
First moment (m)
Second moment (v)
```

For a 7B model:

```text
7B parameters × multiple copies
```

becomes very expensive.

---

# 5. FP32 optimizer states

Even when the model uses FP16/BF16, optimizer states may use FP32 for numerical stability.

Each FP32 value:

```text
4 bytes
```

For 7B parameters:

```text
7B × 4 bytes
≈ 28 GB
```

Adam has two states:

```text
m ≈ 28 GB
v ≈ 28 GB
```

Therefore:

```text
Optimizer states ≈ 56 GB
```

Now the memory looks like:

```text
FP16/BF16 model weights     ≈ 14 GB
Gradients                   ≈ 14 GB
Adam first moment           ≈ 28 GB
Adam second moment          ≈ 28 GB
----------------------------------------
Subtotal                     ≈ 84 GB
```

---

# 6. Master weights may require another copy

Mixed-precision training often maintains a higher-precision copy of weights.

Conceptually:

```text
FP16 model weights
       +
FP32 master weights
```

Why?

Because very small updates can lose precision in FP16.

So:

```text
7B × 4 bytes
≈ 28 GB
```

may be required for FP32 master weights.

Now:

```text
Model weights              ≈ 14 GB
Gradients                  ≈ 14 GB
FP32 master weights        ≈ 28 GB
Adam m                     ≈ 28 GB
Adam v                     ≈ 28 GB
-----------------------------------
Total                      ≈ 112 GB
```

This is before activations.

> Exact memory behavior depends on the framework, precision mode, optimizer, and distributed-training implementation. Modern BF16 training may avoid a separate FP32 master-weight copy in some setups, so this is a useful upper-bound-style illustration rather than a universal formula.

---

# 7. Activations require memory

This is another major component.

During the forward pass:

```text
Input
  ↓
Layer 1 output
  ↓
Layer 2 output
  ↓
Layer 3 output
  ↓
...
```

Normally, many intermediate values must be saved.

Why?

Because during backpropagation:

```text
Backward pass
     ↑
Needs intermediate values
     ↑
Forward pass activations
```

Example:

```python
import torch


x = torch.randn(
    1024,
    1024,
    requires_grad=True
)

y = x * 2

z = torch.relu(y)

loss = z.mean()

loss.backward()
```

The computation graph conceptually keeps information needed for:

```text
loss
 ↑
 z
 ↑
 y
 ↑
 x
```

For an LLM, you have:

```text
Batch size
×
Sequence length
×
Hidden size
×
Number of layers
```

This can become huge.

---

# 8. Sequence length increases activation memory

Suppose:

```text
Batch size = 4

Sequence length = 512
```

Tokens processed:

```text
4 × 512 = 2,048 tokens
```

Now:

```text
Batch size = 4

Sequence length = 4096
```

Tokens processed:

```text
4 × 4096 = 16,384 tokens
```

That's:

```text
8× more tokens
```

So activation memory can increase dramatically.

Simplified:

```text
Activation Memory
≈

Batch Size
×
Sequence Length
×
Hidden Dimension
×
Number of Layers
×
Bytes
```

For transformers, attention-related tensors can also have worse scaling characteristics with sequence length, especially in non-memory-efficient attention implementations.

---

# 9. Example memory estimator

Here is a simplified Python estimator.

```python
def bytes_to_gb(value):

    return value / (
        1024 ** 3
    )


def estimate_full_finetuning_memory(
    num_parameters,
    use_fp32_master_weights=True
):

    # BF16 / FP16 model
    weights = (
        num_parameters * 2
    )

    # Approximate gradients
    gradients = (
        num_parameters * 2
    )

    # Adam optimizer states
    adam_m = (
        num_parameters * 4
    )

    adam_v = (
        num_parameters * 4
    )

    total = (
        weights
        + gradients
        + adam_m
        + adam_v
    )

    if use_fp32_master_weights:

        master_weights = (
            num_parameters * 4
        )

        total += master_weights

    return {

        "weights_gb":
            bytes_to_gb(weights),

        "gradients_gb":
            bytes_to_gb(gradients),

        "adam_m_gb":
            bytes_to_gb(adam_m),

        "adam_v_gb":
            bytes_to_gb(adam_v),

        "total_gb":
            bytes_to_gb(total)
    }
```

Use it:

```python
result = (
    estimate_full_finetuning_memory(
        num_parameters=
            7_000_000_000
    )
)

for name, value in result.items():

    print(
        f"{name}: "
        f"{value:.2f} GB"
    )
```

Approximate result:

```text
weights_gb: 13.04 GB
gradients_gb: 13.04 GB
adam_m_gb: 26.08 GB
adam_v_gb: 26.08 GB
total_gb: 104.31 GB
```

Again, this does **not include activations and runtime overhead**.

---

# 10. Activations make the real requirement even larger

Suppose:

```text
Model/gradients/optimizer ≈ 100 GB
```

Then add:

```text
Activations
Temporary tensors
Attention buffers
CUDA memory
NCCL buffers
Memory fragmentation
```

You could end up with:

```text
120 GB+
```

depending on the training configuration.

---

# 11. See the difference with PyTorch

## Full fine-tuning

```python
model = load_model()

for parameter in model.parameters():

    parameter.requires_grad = True
```

Now every parameter participates in:

```text
Forward
↓
Gradient calculation
↓
Optimizer state
↓
Weight update
```

Example:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)
```

The optimizer maintains state for millions or billions of parameters.

---

# 12. Compare with LoRA

With LoRA:

```python
for parameter in model.parameters():

    parameter.requires_grad = False
```

Then attach LoRA layers:

```text
Original weight:

W

Frozen
```

LoRA learns:

```text
W + ΔW

where:

ΔW = B × A
```

Only:

```text
A
B
```

are trainable.

Example:

```python
from peft import (
    LoraConfig,
    get_peft_model
)


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[

        "q_proj",

        "v_proj"
    ],

    task_type="CAUSAL_LM"
)


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
Full Fine-Tuning

7,000,000,000
trainable parameters
```

versus:

```text
LoRA

7,000,000,000
total parameters

20,000,000
trainable parameters
```

So optimizer states are created for dramatically fewer parameters.

---

# 13. Memory comparison

```text
FULL FINE-TUNING

7B Model

Model Weights      ████████
Gradients          ████████
Optimizer State    ████████████████
Master Weights     ████████████████
Activations        ███████

Total → Very Large
```

```text
LORA

7B Model

Frozen Base        ████████
LoRA Weights       ▏
LoRA Gradients     ▏
LoRA Optimizer     ▏
Activations        ███████

Total → Much Smaller
```

Important: **LoRA still needs the base model and activations for forward/backward computation**, so LoRA does not make memory usage negligible. It mainly saves memory from gradients and optimizer states for the frozen base weights.

---

# 14. How gradient checkpointing helps

Normally:

```text
Forward Pass

Layer 1 activation → STORE
Layer 2 activation → STORE
Layer 3 activation → STORE
Layer 4 activation → STORE
```

Gradient checkpointing changes this:

```text
Forward Pass

Layer 1 activation → STORE checkpoint

Layer 2 → discard

Layer 3 → discard

Layer 4 → STORE checkpoint
```

During backward:

```text
Need Layer 2 activation?

Recompute it.
```

Code:

```python
model.gradient_checkpointing_enable()
```

Or in Transformers:

```python
from transformers import (
    TrainingArguments
)


training_args = TrainingArguments(

    output_dir="./output",

    gradient_checkpointing=True
)
```

Tradeoff:

```text
Memory ↓

Compute time ↑
```

---

# 15. How QLoRA reduces memory further

QLoRA:

```text
7B Base Model
       │
       ▼
4-bit Quantization
       │
       ▼
Frozen
       │
       +
       │
LoRA Adapters
       │
       ▼
Train only adapters
```

Example:

```python
from transformers import (
    BitsAndBytesConfig
)

import torch


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=
        torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

Then:

```python
model = (
    AutoModelForCausalLM
    .from_pretrained(

        MODEL_NAME,

        quantization_config=
            bnb_config,

        device_map="auto"
    )
)
```

Instead of storing base weights roughly as:

```text
BF16 → ~14 GB
```

QLoRA can reduce the base-weight footprint substantially.

---

# 16. The complete memory picture

```text
                    FULL FINE-TUNING

                 ┌───────────────────┐
                 │ Base Model        │
                 │ ~14 GB            │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Gradients         │
                 │ ~14 GB            │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Adam m            │
                 │ ~28 GB            │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Adam v            │
                 │ ~28 GB            │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Master Weights*   │
                 │ ~28 GB            │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Activations       │
                 │ Depends on setup  │
                 └───────────────────┘

                 *implementation-dependent
```

---

# 17. Interview-ready answer

> **Full fine-tuning requires a large amount of GPU memory because every model parameter is trainable. The GPU must store the model weights, gradients for all parameters, optimizer states such as Adam's first and second moments, and activations required for backpropagation. Depending on the precision and implementation, there may also be higher-precision copies of the weights.**
>
> **For a 7B model, BF16 weights alone are roughly 13–14 GB. Gradients require another similar amount, while Adam's optimizer states can require roughly two FP32 values per parameter, adding around 52–56 GB. Activations and CUDA overhead add more memory, so full fine-tuning can require well over 100 GB.**
>
> **To reduce memory, I would use LoRA or QLoRA, mixed precision, gradient checkpointing, FlashAttention or other memory-efficient attention, smaller micro-batches with gradient accumulation, 8-bit optimizers, and distributed approaches such as FSDP or DeepSpeed ZeRO.**

## The key point to remember

```text
Inference:

Model weights
+
small runtime memory
```

```text
Full fine-tuning:

Model weights
+
gradients
+
optimizer states
+
activations
+
temporary buffers
```

That is why **training an LLM requires much more GPU memory than simply running inference with the same model**.
