# What is FSDP?

**FSDP (Fully Sharded Data Parallel)** is a distributed training technique in PyTorch that allows you to train models that may be too large to fit completely on a single GPU.

The key idea is:

> Instead of storing a complete copy of the model, gradients, and optimizer states on every GPU, FSDP **shards (splits) them across multiple GPUs**.

---

# 1. The problem with DDP

In Distributed Data Parallelism (DDP):

```text
GPU 0                     GPU 1

Full Model                Full Model
7B parameters             7B parameters

Full Gradients            Full Gradients

Full Optimizer            Full Optimizer
```

If a model requires:

```text
80 GB GPU memory
```

and you have:

```text
4 × 24 GB GPUs
```

DDP does not solve the problem because each GPU still needs:

```text
80 GB
```

Each GPU has only:

```text
24 GB
```

So:

```text
DDP → ❌ Model doesn't fit
```

---

# 2. How FSDP solves this

FSDP splits the model across GPUs.

Suppose we have:

```text
Model = 80 GB
```

and:

```text
4 GPUs
```

Conceptually:

```text
FSDP

GPU 0 → Model shard 1
GPU 1 → Model shard 2
GPU 2 → Model shard 3
GPU 3 → Model shard 4
```

Instead of:

```text
80 GB per GPU
```

each GPU might store approximately:

```text
80 / 4 = 20 GB
```

plus activations and temporary communication buffers.

So:

```text
4 × 24 GB GPUs
```

can potentially train a model that would not fit under ordinary DDP.

The actual memory is more complicated because FSDP temporarily gathers parameters during computation.

---

# 3. What does FSDP shard?

FSDP can shard:

```text
1. Model parameters
2. Gradients
3. Optimizer states
```

This is the most important concept.

## DDP

```text
GPU 0

Parameters:      FULL
Gradients:       FULL
Optimizer State: FULL
```

```text
GPU 1

Parameters:      FULL
Gradients:       FULL
Optimizer State: FULL
```

---

## FSDP

```text
GPU 0

Parameters:      SHARD 1
Gradients:       SHARD 1
Optimizer State: SHARD 1
```

```text
GPU 1

Parameters:      SHARD 2
Gradients:       SHARD 2
Optimizer State: SHARD 2
```

Conceptually:

```text
                    Full Model

        ┌───────────────┼───────────────┐

        ▼               ▼               ▼

      GPU 0           GPU 1           GPU 2

   Parameters 1     Parameters 2     Parameters 3

   Gradients 1      Gradients 2      Gradients 3

   Optimizer 1      Optimizer 2      Optimizer 3
```

---

# 4. How does training work?

This is where FSDP becomes interesting.

Suppose a model has:

```text
Layer 1
Layer 2
Layer 3
Layer 4
```

And the parameters are distributed:

```text
GPU 0 → Part of model parameters
GPU 1 → Part of model parameters
```

During computation, FSDP needs the full parameters for the layer currently being executed.

The process is conceptually:

```text
Parameter shards
       │
       ▼
   ALL-GATHER
       │
       ▼
Full parameters for current layer
       │
       ▼
Forward pass
       │
       ▼
Free full parameters
       │
       ▼
Move to next layer
```

During backward:

```text
Parameter shards
       │
       ▼
   ALL-GATHER
       │
       ▼
Backward computation
       │
       ▼
   REDUCE-SCATTER
       │
       ▼
Each GPU keeps only its gradient shard
```

This is a simplified conceptual view.

---

# 5. FSDP communication operations

There are two important operations.

## All-Gather

Suppose:

```text
GPU 0 → [A]

GPU 1 → [B]

GPU 2 → [C]
```

After all-gather:

```text
GPU 0 → [A, B, C]

GPU 1 → [A, B, C]

GPU 2 → [A, B, C]
```

FSDP temporarily reconstructs the required parameters.

---

## Reduce-Scatter

Suppose every GPU computes gradients.

FSDP:

```text
Gradients
    │
    ▼
Combine / Reduce
    │
    ▼
Split across GPUs
```

Example:

```text
GPU 0 keeps → Gradient shard A

GPU 1 keeps → Gradient shard B

GPU 2 keeps → Gradient shard C
```

This avoids permanently storing the complete gradient on every GPU.

---

# 6. DDP vs FSDP visual comparison

## DDP

```text
GPU 0                    GPU 1

┌───────────────┐        ┌───────────────┐
│ Full Model    │        │ Full Model    │
│ Full Gradient │        │ Full Gradient │
│ Full Optimizer│        │ Full Optimizer│
└───────────────┘        └───────────────┘
```

## FSDP

```text
GPU 0                    GPU 1

┌───────────────┐        ┌───────────────┐
│ Model Part 1  │        │ Model Part 2  │
│ Gradient 1    │        │ Gradient 2    │
│ Optimizer 1   │        │ Optimizer 2   │
└───────────────┘        └───────────────┘
```

---

# 7. Simple PyTorch FSDP code

Here is a basic example.

## Imports

```python
import os

import torch
import torch.nn as nn

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)
```

---

## Initialize distributed training

```python
def setup_distributed():

    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

    return local_rank
```

---

## Create a model

```python
class SimpleModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1024,
                4096
            ),

            nn.ReLU(),

            nn.Linear(
                4096,
                1024
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)
```

---

## Wrap the model with FSDP

```python
local_rank = setup_distributed()

model = SimpleModel()

model = model.to(
    local_rank
)

model = FSDP(
    model
)
```

The important line is:

```python
model = FSDP(model)
```

Conceptually, FSDP manages:

```text
Parameter sharding
Gradient sharding
Optimizer-state sharding
Communication
```

---

# 8. Complete training loop

```python
import os

import torch
import torch.nn as nn

import torch.optim as optim

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from torch.utils.data.distributed import (
    DistributedSampler
)
```

## Setup

```python
def setup():

    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

    return local_rank
```

## Model

```python
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                100,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                10
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)
```

## Training

```python
def train():

    local_rank = setup()


    # Model

    model = Model()

    model = model.to(
        local_rank
    )


    # Wrap with FSDP

    model = FSDP(
        model
    )


    # Optimizer

    optimizer = optim.AdamW(

        model.parameters(),

        lr=1e-3
    )


    loss_function = nn.MSELoss()


    # Dataset

    x = torch.randn(
        10_000,
        100
    )

    y = torch.randn(
        10_000,
        10
    )


    dataset = TensorDataset(
        x,
        y
    )


    sampler = DistributedSampler(
        dataset
    )


    dataloader = DataLoader(

        dataset,

        batch_size=32,

        sampler=sampler
    )


    # Training loop

    for epoch in range(10):

        sampler.set_epoch(
            epoch
        )


        for x_batch, y_batch in dataloader:

            x_batch = x_batch.to(
                local_rank
            )

            y_batch = y_batch.to(
                local_rank
            )


            optimizer.zero_grad()


            # Forward

            predictions = model(
                x_batch
            )


            # Loss

            loss = loss_function(

                predictions,

                y_batch
            )


            # Backward

            loss.backward()


            # Optimizer update

            optimizer.step()


        if local_rank == 0:

            print(
                f"Epoch {epoch} "
                f"Loss: {loss.item()}"
            )


    dist.destroy_process_group()
```

Run:

```python
if __name__ == "__main__":

    train()
```

Launch:

```bash
torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_fsdp.py
```

This creates:

```text
Process 0 → GPU 0
Process 1 → GPU 1
Process 2 → GPU 2
Process 3 → GPU 3
```

---

# 9. FSDP for an LLM

For LLMs, we normally use **auto wrapping**.

Why?

An LLM has many repeated transformer blocks:

```text
Embedding
    │
    ▼
Transformer Block 1
    │
    ▼
Transformer Block 2
    │
    ▼
Transformer Block 3
    │
    ▼
...
    │
    ▼
Transformer Block N
```

Instead of treating the entire model as one huge FSDP unit, we often wrap transformer blocks individually.

---

## Example auto-wrap policy

```python
from functools import partial

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)
```

Suppose your model has:

```python
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer
)
```

Create a wrapping policy:

```python
auto_wrap_policy = partial(

    transformer_auto_wrap_policy,

    transformer_layer_cls={
        LlamaDecoderLayer
    }
)
```

Then:

```python
model = FSDP(

    model,

    auto_wrap_policy=
        auto_wrap_policy
)
```

Conceptually:

```text
FSDP

Embedding

FSDP(
    Transformer Block 1
)

FSDP(
    Transformer Block 2
)

FSDP(
    Transformer Block 3
)

...

Output Layer
```

This allows FSDP to:

```text
Gather one transformer block
      ↓
Compute
      ↓
Release memory
      ↓
Gather next block
```

This is much more memory efficient than materializing the entire model.

---

# 10. Mixed precision with FSDP

FSDP can use BF16.

```python
from torch.distributed.fsdp import (
    MixedPrecision
)
```

Configure:

```python
mixed_precision = MixedPrecision(

    param_dtype=torch.bfloat16,

    reduce_dtype=torch.bfloat16,

    buffer_dtype=torch.bfloat16
)
```

Use:

```python
model = FSDP(

    model,

    auto_wrap_policy=
        auto_wrap_policy,

    mixed_precision=
        mixed_precision
)
```

This reduces memory and can improve performance on supported GPUs.

---

# 11. Activation checkpointing

Even with parameter sharding, activations can consume a lot of memory.

Example:

```text
FSDP reduces:

✓ Parameters
✓ Gradients
✓ Optimizer states
```

But activations can still be large.

So we combine:

```text
FSDP
+
Activation Checkpointing
```

Conceptually:

```text
Forward pass:

Layer 1 → Save checkpoint

Layer 2 → Discard activations

Layer 3 → Discard activations
```

During backward:

```text
Need activation?

Recompute it
```

Tradeoff:

```text
GPU Memory ↓

Training Compute ↑
```

For Hugging Face models:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

---

# 12. CPU offloading

FSDP can optionally move some state to CPU.

Conceptually:

```text
GPU Memory

Current layer
Current activations
```

while some parameters may be offloaded:

```text
CPU RAM

Other model data
```

Example:

```python
from torch.distributed.fsdp import (
    CPUOffload
)


cpu_offload = CPUOffload(
    offload_params=True
)


model = FSDP(

    model,

    cpu_offload=cpu_offload
)
```

This can reduce GPU memory but may slow training because of:

```text
CPU ↔ GPU transfer
```

---

# 13. FSDP sharding strategies

FSDP supports different strategies.

A common one is:

```python
from torch.distributed.fsdp import (
    ShardingStrategy
)
```

Example:

```python
model = FSDP(

    model,

    sharding_strategy=
        ShardingStrategy.FULL_SHARD
)
```

`FULL_SHARD` conceptually means:

```text
Parameters       → Sharded
Gradients        → Sharded
Optimizer states → Sharded
```

This provides strong memory savings.

Another strategy:

```text
SHARD_GRAD_OP
```

may keep parameters replicated while sharding some other states, depending on the configuration.

---

# 14. FSDP vs DDP memory example

Suppose:

```text
Model weights      = 20 GB
Gradients          = 20 GB
Optimizer states   = 40 GB

Total training state = 80 GB
```

You have:

```text
4 GPUs
```

## DDP

Every GPU stores:

```text
GPU 0 → 80 GB

GPU 1 → 80 GB

GPU 2 → 80 GB

GPU 3 → 80 GB
```

Not possible on:

```text
4 × 24 GB GPUs
```

---

## FSDP

Conceptually:

```text
80 GB / 4 GPUs
```

approximately:

```text
20 GB per GPU
```

plus:

```text
Activations
Temporary gathered parameters
Communication buffers
```

So FSDP can potentially fit, though you cannot simply divide total memory by GPU count when sizing a real job.

---

# 15. DDP vs FSDP vs ZeRO

| Feature                  | DDP                | FSDP         | DeepSpeed ZeRO       |
| ------------------------ | ------------------ | ------------ | -------------------- |
| Parameters sharded       | ❌                  | ✅            | ZeRO stage dependent |
| Gradients sharded        | ❌                  | ✅            | Stage dependent      |
| Optimizer states sharded | ❌                  | ✅            | Stage dependent      |
| Reduces per-GPU memory   | Limited            | Strongly     | Strongly             |
| Model replicated         | Yes                | Sharded      | Stage dependent      |
| Complexity               | Lower              | Medium       | Medium/High          |
| Best for                 | Model fits per GPU | Large models | Large-scale training |

A useful mapping is:

```text
DDP
≈ replicate model

FSDP FULL_SHARD
≈ shard parameters + gradients + optimizer states

ZeRO
≈ progressively shard optimizer → gradients → parameters
```

The exact implementation details differ, but this is a useful interview-level comparison.

---

# 16. FSDP + LLM fine-tuning architecture

For a large LLM:

```text
                    Training Cluster
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼

      GPU 0              GPU 1              GPU 2

   Param shard A      Param shard B      Param shard C

   Grad shard A       Grad shard B       Grad shard C

   Optimizer A        Optimizer B        Optimizer C

        │                  │                  │

        └──────── Communication ─────────────┘

                   All-Gather
                   Reduce-Scatter
```

Each GPU also processes different training samples.

Therefore FSDP combines:

```text
Data parallelism
+
State sharding
```

This is why it is powerful for large model training.

---

# 17. Production-style configuration concept

A typical LLM training setup might look like:

```text
Large LLM
   +
FSDP FULL_SHARD
   +
BF16
   +
Transformer block auto-wrap
   +
Activation checkpointing
   +
Gradient accumulation
```

Conceptually:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False


model = FSDP(

    model,

    auto_wrap_policy=
        auto_wrap_policy,

    mixed_precision=
        mixed_precision,

    sharding_strategy=
        ShardingStrategy.FULL_SHARD
)
```

This combination addresses different memory sources:

| Technique             | Reduces                             |
| --------------------- | ----------------------------------- |
| FSDP                  | Parameter/gradient/optimizer memory |
| BF16                  | Numeric tensor memory               |
| Checkpointing         | Activation memory                   |
| Gradient accumulation | Micro-batch memory                  |
| CPU offloading        | GPU memory at performance cost      |

---

# 18. Important limitation of FSDP

FSDP reduces memory, but increases:

```text
Communication
Synchronization
Implementation complexity
```

During training:

```text
GPU 0 needs layer parameters
        │
        ▼
ALL-GATHER
        │
        ▼
Compute
        │
        ▼
Release / Reshard
```

Therefore performance depends heavily on:

```text
GPU interconnect
NVLink
InfiniBand
Network bandwidth
Model architecture
Batch size
```

So FSDP is not automatically faster than DDP.

Its primary advantage is:

> **Making larger models trainable by reducing per-GPU memory usage.**

---

# Interview-ready answer

> **FSDP, or Fully Sharded Data Parallel, is a PyTorch distributed training technique designed to reduce per-GPU memory usage for large models. Unlike DDP, which replicates the full model, gradients, and optimizer states on every GPU, FSDP shards these states across multiple GPUs.**
>
> **During computation, FSDP temporarily all-gathers the parameters needed for the current module, performs the forward or backward computation, and then reshardes them. Gradients are typically synchronized using reduce-scatter so each GPU retains only its shard.**
>
> **For LLM training, I would typically combine FSDP with BF16 mixed precision, transformer-block auto-wrapping, activation checkpointing, and gradient accumulation. FSDP helps when the model does not fit on a single GPU, whereas DDP is primarily useful when the full model already fits and the goal is higher throughput.**

## Easiest way to remember

```text
DDP:

GPU 0 → Full Model
GPU 1 → Full Model
GPU 2 → Full Model

Good for speed
```

```text
FSDP:

GPU 0 → Model Part A
GPU 1 → Model Part B
GPU 2 → Model Part C

Good for memory
```

### One-line distinction

> **DDP replicates the training state; FSDP shards the training state.**
