# What is ZeRO?

**ZeRO (Zero Redundancy Optimizer)** is a distributed training technique, originally introduced in Microsoft DeepSpeed, that reduces GPU memory usage by **removing redundant copies of training state across GPUs**.

The core problem is that normal data-parallel training duplicates a lot of memory.

## Normal DDP

With 4 GPUs:

```text
GPU 0                 GPU 1

Full Parameters       Full Parameters
Full Gradients        Full Gradients
Full Optimizer        Full Optimizer


GPU 2                 GPU 3

Full Parameters       Full Parameters
Full Gradients        Full Gradients
Full Optimizer        Full Optimizer
```

Every GPU stores almost the same training state.

ZeRO asks:

> Why should all GPUs store identical optimizer states, gradients, and parameters?

It distributes that state across GPUs instead.

---

# 1. What consumes memory during training?

For a large LLM, GPU memory is mainly consumed by:

```text
1. Model parameters
2. Gradients
3. Optimizer states
4. Activations
5. Temporary buffers
```

For example:

```text
7B LLM
│
├── Parameters
├── Gradients
├── Adam optimizer states
├── Activations
└── Temporary computation memory
```

ZeRO primarily reduces the memory used by:

```text
✓ Optimizer states
✓ Gradients
✓ Parameters
```

depending on the ZeRO stage.

---

# 2. Why is it called "Zero Redundancy"?

Suppose we have:

```text
4 GPUs
```

With traditional DDP:

```text
GPU 0 → Full optimizer state
GPU 1 → Full optimizer state
GPU 2 → Full optimizer state
GPU 3 → Full optimizer state
```

This means the optimizer state is duplicated four times.

ZeRO partitions the state:

```text
GPU 0 → Optimizer shard 1
GPU 1 → Optimizer shard 2
GPU 2 → Optimizer shard 3
GPU 3 → Optimizer shard 4
```

So unnecessary duplication is reduced.

Hence:

```text
Zero Redundancy Optimizer
```

---

# 3. ZeRO has three main stages

```text
ZeRO Stage 1
      ↓
Shard optimizer states

ZeRO Stage 2
      ↓
Shard optimizer states
+
Shard gradients

ZeRO Stage 3
      ↓
Shard optimizer states
+
Shard gradients
+
Shard parameters
```

Let's understand each properly.

---

# 4. ZeRO Stage 1

## What does it shard?

```text
Optimizer states only
```

Suppose we have 2 GPUs.

### Traditional DDP

```text
GPU 0:

Parameters      FULL
Gradients       FULL
Optimizer       FULL
```

```text
GPU 1:

Parameters      FULL
Gradients       FULL
Optimizer       FULL
```

### ZeRO Stage 1

```text
GPU 0:

Parameters      FULL
Gradients       FULL
Optimizer       PART A
```

```text
GPU 1:

Parameters      FULL
Gradients       FULL
Optimizer       PART B
```

The model and gradients are still replicated.

Only optimizer state is partitioned.

---

## Why does Stage 1 help?

Optimizers such as Adam maintain extra state.

Conceptually:

```text
Parameter
    │
    ├── First moment
    │
    └── Second moment
```

For every model parameter, Adam stores additional tensors.

Therefore:

```text
Large model
     ↓
Large optimizer memory
```

ZeRO Stage 1 distributes this memory.

---

# 5. ZeRO Stage 2

ZeRO Stage 2 shards:

```text
✓ Optimizer states
✓ Gradients
```

### GPU 0

```text
Parameters      FULL
Gradients       PART A
Optimizer       PART A
```

### GPU 1

```text
Parameters      FULL
Gradients       PART B
Optimizer       PART B
```

Conceptually:

```text
GPU 0              GPU 1

Full Model         Full Model

Gradient A         Gradient B

Optimizer A        Optimizer B
```

This gives more memory savings than Stage 1.

---

# 6. ZeRO Stage 3

This is the most memory-efficient standard ZeRO stage.

It shards:

```text
✓ Optimizer states
✓ Gradients
✓ Parameters
```

### GPU 0

```text
Parameters      PART A
Gradients       PART A
Optimizer       PART A
```

### GPU 1

```text
Parameters      PART B
Gradients       PART B
Optimizer       PART B
```

### GPU 2

```text
Parameters      PART C
Gradients       PART C
Optimizer       PART C
```

Conceptually:

```text
                    Full LLM

       ┌──────────────┼──────────────┐

       ▼              ▼              ▼

     GPU 0          GPU 1          GPU 2

    Model A        Model B        Model C

    Grad A         Grad B         Grad C

    Opt A          Opt B          Opt C
```

This is useful when:

> **The model cannot fit entirely on a single GPU.**

---

# 7. How does ZeRO-3 actually train if each GPU has only part of the model?

This is the most important concept.

Suppose your model is:

```text
Layer 1
Layer 2
Layer 3
Layer 4
```

Parameters are sharded.

When GPU computation needs a layer:

```text
Parameter shards
      │
      ▼
   All-Gather
      │
      ▼
Reconstruct required parameters
      │
      ▼
Forward computation
      │
      ▼
Release / partition parameters
```

For gradients:

```text
Local gradient computation
         │
         ▼
    Reduce operation
         │
         ▼
    Reduce-Scatter
         │
         ▼
Each GPU keeps only its
gradient shard
```

So the model is not permanently fully stored on every GPU.

---

# 8. ZeRO communication operations

## All-Gather

Suppose:

```text
GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
```

After:

```text
All-Gather
```

each GPU temporarily has:

```text
[A, B, C]
```

This can reconstruct parameters needed for computation.

---

## Reduce-Scatter

Suppose all GPUs produce gradients.

Conceptually:

```text
GPU 0 gradients ──┐
GPU 1 gradients ──┤
GPU 2 gradients ──┘
                    ↓
                 Reduce
                    ↓
                 Scatter
                    ↓

GPU 0 → Gradient shard A
GPU 1 → Gradient shard B
GPU 2 → Gradient shard C
```

This prevents every GPU from permanently storing the complete gradient.

---

# 9. ZeRO stages comparison

| Training state           | DDP        | ZeRO-1      | ZeRO-2      | ZeRO-3      |
| ------------------------ | ---------- | ----------- | ----------- | ----------- |
| Parameters               | Replicated | Replicated  | Replicated  | **Sharded** |
| Gradients                | Replicated | Replicated  | **Sharded** | **Sharded** |
| Optimizer states         | Replicated | **Sharded** | **Sharded** | **Sharded** |
| Memory savings           | Low        | Good        | Better      | Highest     |
| Communication complexity | Low        | Medium      | Medium      | Higher      |

The easiest way to remember:

```text
Stage 1
Optimizer

Stage 2
Optimizer + Gradients

Stage 3
Optimizer + Gradients + Parameters
```

---

# 10. Memory example

Suppose a model requires:

```text
Parameters       = 20 GB
Gradients        = 20 GB
Optimizer states = 40 GB
```

Total:

```text
80 GB
```

Assume:

```text
4 GPUs
```

## DDP

Every GPU approximately needs:

```text
GPU 0 → 80 GB
GPU 1 → 80 GB
GPU 2 → 80 GB
GPU 3 → 80 GB
```

---

## ZeRO-1

Optimizer state is distributed:

```text
Parameters = 20 GB
Gradients  = 20 GB
Optimizer  = 40 / 4 = 10 GB
```

Approximately:

```text
50 GB per GPU
```

---

## ZeRO-2

```text
Parameters = 20 GB
Gradients  = 20 / 4 = 5 GB
Optimizer  = 40 / 4 = 10 GB
```

Approximately:

```text
35 GB per GPU
```

---

## ZeRO-3

```text
Parameters = 20 / 4 = 5 GB
Gradients  = 20 / 4 = 5 GB
Optimizer  = 40 / 4 = 10 GB
```

Approximately:

```text
20 GB per GPU
```

This is simplified.

Real memory also includes:

```text
Activations
Temporary buffers
Communication buffers
CUDA memory fragmentation
```

So you should **never assume memory is exactly divided by the number of GPUs**.

---

# 11. ZeRO-1 configuration

Example DeepSpeed configuration:

```json
{
    "train_batch_size": 32,

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 1
    }
}
```

---

# 12. ZeRO-2 configuration

```json
{
    "train_batch_size": 32,

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 2
    }
}
```

---

# 13. ZeRO-3 configuration

```json
{
    "train_batch_size": 32,

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 3
    }
}
```

---

# 14. Using ZeRO with Hugging Face

Create:

```text
ds_zero3.json
```

```json
{
    "bf16": {
        "enabled": "auto"
    },

    "zero_optimization": {

        "stage": 3,

        "overlap_comm": true,

        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",

    "train_micro_batch_size_per_gpu": "auto"
}
```

Then:

```python
from transformers import TrainingArguments


training_args = TrainingArguments(

    output_dir="./output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-5,

    bf16=True,

    deepspeed="./ds_zero3.json"
)
```

And use `Trainer`:

```python
from transformers import Trainer


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset
)


trainer.train()
```

Hugging Face integrates with DeepSpeed, which then manages ZeRO according to the configuration.

---

# 15. ZeRO with LoRA fine-tuning

For LoRA:

```text
Base Model
     │
     ▼
Mostly Frozen
     │
     +
LoRA Adapters
     │
     ▼
Only small adapter parameters train
```

Memory requirements are already lower.

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

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


model = get_peft_model(

    model,

    lora_config
)
```

For a moderately sized model:

```text
QLoRA
+
Gradient Checkpointing
```

may be sufficient.

For very large models or multi-GPU training:

```text
LoRA
+
DeepSpeed ZeRO
```

can still be useful.

---

# 16. ZeRO vs FSDP

This is an important interview question.

## ZeRO Stage 3

```text
GPU 0

Parameter shard
Gradient shard
Optimizer shard
```

```text
GPU 1

Parameter shard
Gradient shard
Optimizer shard
```

## FSDP

```text
GPU 0

Parameter shard
Gradient shard
Optimizer shard
```

```text
GPU 1

Parameter shard
Gradient shard
Optimizer shard
```

They are conceptually similar at a high level.

A useful mental model:

```text
FSDP FULL_SHARD
        ≈
ZeRO Stage 3
```

Both reduce memory by sharding training state.

But they differ in:

```text
Implementation
APIs
Runtime behavior
Wrapping strategies
Ecosystem integration
Offloading capabilities
```

---

# 17. ZeRO-Offload

What if GPU memory is still insufficient?

ZeRO can offload some state to CPU memory.

Example:

```json
{
    "zero_optimization": {

        "stage": 2,

        "offload_optimizer": {

            "device": "cpu",

            "pin_memory": true
        }
    }
}
```

Conceptually:

```text
GPU

Model
Gradients
Forward/Backward computation
```

```text
CPU

Optimizer states
```

This gives:

```text
GPU Memory ↓
```

but:

```text
CPU ↔ GPU transfer ↑
```

So training may become slower.

---

# 18. ZeRO-Infinity

For extremely large models, DeepSpeed can use a hierarchy:

```text
Fastest

GPU
 │
 ▼

CPU RAM
 │
 ▼

NVMe Storage

Slowest
```

Conceptually:

```text
GPU Memory insufficient
        │
        ▼
CPU Offloading
        │
Still insufficient?
        │
        ▼
NVMe Offloading
```

This makes it possible to work with models much larger than GPU memory, although storage transfers can significantly affect throughput.

---

# 19. When should you use each ZeRO stage?

## ZeRO Stage 1

Use when:

```text
Model fits on each GPU
But optimizer states consume too much memory
```

---

## ZeRO Stage 2

Use when:

```text
Model fits
But gradients + optimizer states are too large
```

Often a good balance between:

```text
Memory savings
+
Performance
```

---

## ZeRO Stage 3

Use when:

```text
Model itself does not fit on one GPU
```

Best for:

```text
Large LLM training
Very large models
Memory-constrained GPUs
```

But it introduces more communication overhead.

---

# 20. ZeRO vs DDP

```text
DDP

Each GPU:

Full Model
Full Gradients
Full Optimizer
```

vs:

```text
ZeRO

Training state is distributed
across GPUs
```

Therefore:

> **DDP scales compute by replicating the model. ZeRO scales memory efficiency by removing redundant training state.**

---

# Interview-ready answer

> **ZeRO stands for Zero Redundancy Optimizer. It is a distributed training technology used by DeepSpeed to reduce GPU memory consumption by partitioning training states across multiple GPUs instead of replicating them on every GPU.**
>
> **ZeRO Stage 1 shards optimizer states, Stage 2 shards optimizer states and gradients, and Stage 3 shards optimizer states, gradients, and model parameters.**
>
> **ZeRO-3 is especially useful for training very large LLMs because the complete model state does not need to permanently reside on every GPU. During computation, the required parameters are temporarily gathered, and gradients are synchronized and partitioned across GPUs.**
>
> **In practice, I would choose ZeRO-2 when the model fits on each GPU but training state is too large, and ZeRO-3 when the model itself cannot fit on one GPU.**

## One-line memory trick

```text
ZeRO-1 → Optimizer
ZeRO-2 → Optimizer + Gradients
ZeRO-3 → Optimizer + Gradients + Parameters
```

> **ZeRO removes redundant training-state copies across GPUs to make large-model training possible.**
