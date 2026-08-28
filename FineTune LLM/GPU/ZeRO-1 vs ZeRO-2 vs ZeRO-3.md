# ZeRO-1 vs ZeRO-2 vs ZeRO-3

The simplest difference is:

```text
ZeRO-1 → Shards Optimizer States

ZeRO-2 → Shards Optimizer States + Gradients

ZeRO-3 → Shards Optimizer States + Gradients + Parameters
```

ZeRO progressively removes redundant copies of training state across data-parallel GPUs. ([DeepSpeed][1])

---

# 1. First understand the problem: DDP

Suppose you have **4 GPUs**.

In traditional DDP:

```text
GPU 0                GPU 1

Full Parameters      Full Parameters
Full Gradients       Full Gradients
Full Optimizer       Full Optimizer


GPU 2                GPU 3

Full Parameters      Full Parameters
Full Gradients       Full Gradients
Full Optimizer       Full Optimizer
```

Every GPU stores almost everything.

That creates redundant memory usage.

ZeRO progressively removes this redundancy.

---

# 2. ZeRO-1

## What does ZeRO-1 shard?

```text
Optimizer states only
```

### GPU layout

```text
GPU 0

Parameters       FULL
Gradients        FULL
Optimizer        SHARD 1
```

```text
GPU 1

Parameters       FULL
Gradients        FULL
Optimizer        SHARD 2
```

The model parameters and gradients still exist fully on every GPU.

Only optimizer state is distributed.

### Why is this useful?

Adam-like optimizers maintain additional states such as:

```text
Parameter
   │
   ├── First moment (m)
   │
   └── Second moment (v)
```

For a large LLM, optimizer states consume significant memory.

### Configuration

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

---

# 3. ZeRO-2

ZeRO-2 includes everything from ZeRO-1 and additionally shards gradients.

```text
ZeRO-2
=
Optimizer states
+
Gradients
```

### GPU layout

```text
GPU 0

Parameters       FULL
Gradients        SHARD 1
Optimizer        SHARD 1
```

```text
GPU 1

Parameters       FULL
Gradients        SHARD 2
Optimizer        SHARD 2
```

The full model still exists on every GPU.

But gradients and optimizer states are partitioned.

DeepSpeed documents ZeRO-2 as partitioning optimizer and gradient state. ([DeepSpeed][1])

### Configuration

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true
  }
}
```

`contiguous_gradients` can help reduce memory fragmentation during backpropagation.

---

# 4. ZeRO-3

ZeRO-3 is the most aggressive standard ZeRO stage.

```text
ZeRO-3
=
Optimizer states
+
Gradients
+
Parameters
```

Everything important is sharded.

### GPU layout

```text
GPU 0

Parameters       SHARD 1
Gradients        SHARD 1
Optimizer        SHARD 1
```

```text
GPU 1

Parameters       SHARD 2
Gradients        SHARD 2
Optimizer        SHARD 2
```

```text
GPU 2

Parameters       SHARD 3
Gradients        SHARD 3
Optimizer        SHARD 3
```

DeepSpeed states that ZeRO-3 partitions parameters and automatically collects/partitions them during forward and backward computation. ([DeepSpeed][1])

### Configuration

```json
{
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true
  }
}
```

---

# 5. Visual comparison

## DDP

```text
GPU 0          GPU 1          GPU 2

Model FULL     Model FULL     Model FULL

Grad FULL      Grad FULL      Grad FULL

Opt FULL       Opt FULL       Opt FULL
```

---

## ZeRO-1

```text
GPU 0          GPU 1          GPU 2

Model FULL     Model FULL     Model FULL

Grad FULL      Grad FULL      Grad FULL

Opt A          Opt B          Opt C
```

---

## ZeRO-2

```text
GPU 0          GPU 1          GPU 2

Model FULL     Model FULL     Model FULL

Grad A         Grad B         Grad C

Opt A          Opt B          Opt C
```

---

## ZeRO-3

```text
GPU 0          GPU 1          GPU 2

Model A        Model B        Model C

Grad A         Grad B         Grad C

Opt A          Opt B          Opt C
```

---

# 6. The most important comparison table

| Feature                | DDP       | ZeRO-1      | ZeRO-2      | ZeRO-3      |
| ---------------------- | --------- | ----------- | ----------- | ----------- |
| Parameters             | Full copy | Full copy   | Full copy   | **Sharded** |
| Gradients              | Full copy | Full copy   | **Sharded** | **Sharded** |
| Optimizer states       | Full copy | **Sharded** | **Sharded** | **Sharded** |
| Memory savings         | Low       | Medium      | High        | Highest     |
| Communication overhead | Low       | Lower       | Medium      | Higher      |
| Complexity             | Low       | Low         | Medium      | High        |
| Large model support    | Limited   | Better      | Better      | Best        |

---

# 7. Memory example

Suppose your training state is:

```text
Parameters        = 20 GB
Gradients         = 20 GB
Optimizer states  = 40 GB

Total             = 80 GB
```

And you have:

```text
4 GPUs
```

## DDP

Every GPU stores:

```text
20 GB Parameters
20 GB Gradients
40 GB Optimizer

= 80 GB per GPU
```

---

## ZeRO-1

Optimizer is sharded:

```text
Parameters = 20 GB
Gradients  = 20 GB
Optimizer  = 40 / 4 = 10 GB

Total ≈ 50 GB per GPU
```

---

## ZeRO-2

Optimizer + gradients are sharded:

```text
Parameters = 20 GB
Gradients  = 20 / 4 = 5 GB
Optimizer  = 40 / 4 = 10 GB

Total ≈ 35 GB per GPU
```

---

## ZeRO-3

Everything is sharded:

```text
Parameters = 20 / 4 = 5 GB
Gradients  = 20 / 4 = 5 GB
Optimizer  = 40 / 4 = 10 GB

Total ≈ 20 GB per GPU
```

So conceptually:

```text
DDP     ≈ 80 GB/GPU

ZeRO-1  ≈ 50 GB/GPU

ZeRO-2  ≈ 35 GB/GPU

ZeRO-3  ≈ 20 GB/GPU
```

These are **simplified illustrative numbers**. Real training also requires activation memory, communication buffers, temporary tensors, and CUDA overhead.

---

# 8. How does ZeRO-3 work during training?

A common question is:

> If GPU 0 has only part of the model, how can it run the model?

Suppose:

```text
Model

Layer 1
Layer 2
Layer 3
```

During computation:

```text
Parameter shards
       │
       ▼
   ALL-GATHER
       │
       ▼
Required layer parameters
temporarily available
       │
       ▼
Forward computation
       │
       ▼
Parameters partitioned again
```

During backward:

```text
Backward computation
       │
       ▼
Gradients calculated
       │
       ▼
REDUCE-SCATTER
       │
       ▼
Each GPU retains only
its gradient partition
```

This extra communication is why ZeRO-3 saves more memory but can have more communication overhead.

---

# 9. Code: ZeRO-1

```json
{
  "train_micro_batch_size_per_gpu": 4,

  "bf16": {
    "enabled": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00002
    }
  },

  "zero_optimization": {
    "stage": 1
  }
}
```

Use when:

```text
✓ Model fits on every GPU
✓ Optimizer state is the main memory problem
✓ You want relatively simple scaling
```

---

# 10. Code: ZeRO-2

```json
{
  "train_micro_batch_size_per_gpu": 2,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 2,

    "contiguous_gradients": true,

    "reduce_scatter": true
  }
}
```

Use when:

```text
✓ Model fits on every GPU
✓ Gradients + optimizer consume too much memory
✓ You want a good memory/performance balance
```

ZeRO-2 is often considered a useful middle ground.

---

# 11. Code: ZeRO-3

```json
{
  "train_micro_batch_size_per_gpu": 1,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,

    "contiguous_gradients": true,

    "stage3_prefetch_bucket_size": 100000000,

    "stage3_param_persistence_threshold": 100000
  }
}
```

Use when:

```text
✓ Model does NOT fit on one GPU
✓ You need maximum memory savings
✓ You are training a very large LLM
```

DeepSpeed provides additional ZeRO-3 options for parameter gathering, prefetching, persistence, and offloading. ([DeepSpeed][2])

---

# 12. Using ZeRO with Hugging Face Trainer

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "your-model-name"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)
```

Configure training:

```python
training_args = TrainingArguments(
    output_dir="./output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    learning_rate=2e-5,

    bf16=True,

    deepspeed="./zero3_config.json",

    logging_steps=10,

    save_steps=500
)
```

Create the trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

Changing:

```python
deepspeed="./zero1_config.json"
```

to:

```python
deepspeed="./zero2_config.json"
```

or:

```python
deepspeed="./zero3_config.json"
```

changes the ZeRO strategy.

---

# 13. Which one should you choose?

## Use ZeRO-1 when

```text
Model fits on GPU
        ↓
But
        ↓
Optimizer memory is too high
```

Example:

```text
GPU: 24 GB

Model: 10 GB
Gradients: 10 GB
Optimizer: 20 GB

Problem → Optimizer too large

Solution → ZeRO-1
```

---

## Use ZeRO-2 when

```text
Model fits
        ↓
But
        ↓
Gradients + optimizer
consume too much memory
```

This is often a good choice for distributed full fine-tuning.

---

## Use ZeRO-3 when

```text
The model itself
does not fit
on one GPU
```

Example:

```text
GPU memory = 24 GB

Model training state = 80 GB

4 GPUs available
```

ZeRO-3 distributes model state:

```text
GPU 0 → Model shard
GPU 1 → Model shard
GPU 2 → Model shard
GPU 3 → Model shard
```

---

# 14. ZeRO-2 vs ZeRO-3: the most important distinction

This is frequently asked in interviews.

### ZeRO-2

```text
Parameters → Replicated

Gradients → Sharded

Optimizer → Sharded
```

Every GPU still needs the complete model.

---

### ZeRO-3

```text
Parameters → Sharded

Gradients → Sharded

Optimizer → Sharded
```

No GPU needs to permanently store the entire model state.

Therefore:

> **Use ZeRO-2 when the model fits but training states don't fit. Use ZeRO-3 when the model itself doesn't fit on one GPU.**

---

# 15. ZeRO-3 + CPU offloading

If GPU memory is still insufficient:

```json
{
  "zero_optimization": {
    "stage": 3,

    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },

    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

Conceptually:

```text
GPU
│
├── Active computation
├── Active parameters
└── Activations

CPU
│
├── Optimizer states
└── Offloaded parameters
```

DeepSpeed's ZeRO-Infinity capabilities extend offloading to CPU and NVMe for very large models, with the expected trade-off of additional data movement. ([DeepSpeed][1])

---

# Final interview answer

> **ZeRO has three stages for progressively reducing redundant memory in distributed training. ZeRO-1 partitions optimizer states, ZeRO-2 partitions optimizer states and gradients, and ZeRO-3 partitions optimizer states, gradients, and model parameters.**
>
> **The key difference is that ZeRO-1 and ZeRO-2 still replicate the full model on every GPU, while ZeRO-3 shards the model parameters themselves. Therefore, I would use ZeRO-1 when optimizer memory is the bottleneck, ZeRO-2 when optimizer and gradient memory are the bottleneck, and ZeRO-3 when the model itself cannot fit on a single GPU.**

### Best memory trick

```text
ZeRO-1 → Optimizer

ZeRO-2 → Optimizer + Gradients

ZeRO-3 → Optimizer + Gradients + Parameters
```

[1]: https://deepspeed.readthedocs.io/en/stable/zero3.html?utm_source=chatgpt.com "ZeRO — DeepSpeed 0.19.2 documentation"
[2]: https://www.deepspeed.ai/docs/config-json/?utm_source=chatgpt.com "DeepSpeed Configuration JSON - DeepSpeed"
