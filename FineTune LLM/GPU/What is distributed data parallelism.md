# What is Distributed Data Parallelism (DDP)?

**Distributed Data Parallelism** means training the **same model on multiple GPUs**, where each GPU processes a different part of the training data.

The most common PyTorch implementation is:

```text
DistributedDataParallel (DDP)
```

The basic idea:

```text
                    Full Training Dataset
                           │
                ┌──────────┼──────────┐
                │          │          │
                ▼          ▼          ▼
              GPU 0      GPU 1      GPU 2
                │          │          │
              Model      Model      Model
              Copy       Copy       Copy
                │          │          │
              Data A     Data B     Data C
                │          │          │
                ▼          ▼          ▼
             Gradients   Gradients  Gradients
                │          │          │
                └──────────┼──────────┘
                           ▼
                    All-Reduce
                           │
                           ▼
                Same Gradients Everywhere
                           │
                           ▼
                    Update Model
```

---

# 1. Why do we need DDP?

Suppose you have:

```text
1 GPU
```

and your training dataset has:

```text
1,000,000 examples
```

One GPU processes:

```text
Example 1
Example 2
Example 3
...
```

Training can be slow.

With four GPUs:

```text
GPU 0 → Examples 1–250,000
GPU 1 → Examples 250,001–500,000
GPU 2 → Examples 500,001–750,000
GPU 3 → Examples 750,001–1,000,000
```

Now multiple GPUs work simultaneously.

This is called **data parallelism**.

---

# 2. The important point: every GPU has a model copy

With DDP:

```text
GPU 0                    GPU 1

Model A                  Model A
Same architecture        Same architecture
Same weights             Same weights
```

Initially:

```text
W0 = W1
```

Each GPU receives different data.

Example:

```text
GPU 0

Input:
"What is Python?"
```

```text
GPU 1

Input:
"Explain FastAPI"
```

Both GPUs:

```text
Forward pass
     ↓
Calculate loss
     ↓
Backward pass
     ↓
Calculate gradients
```

Then they synchronize gradients.

---

# 3. What is gradient synchronization?

Suppose two GPUs calculate:

```text
GPU 0 gradient:

[2, 4, 6]
```

GPU 1 calculates:

```text
GPU 1 gradient:

[4, 8, 10]
```

DDP performs an **all-reduce** operation.

Average:

```text
[
(2 + 4) / 2,
(4 + 8) / 2,
(6 + 10) / 2
]
```

Result:

```text
[3, 6, 8]
```

Now:

```text
GPU 0 → [3, 6, 8]

GPU 1 → [3, 6, 8]
```

Both GPUs use the same gradient.

Then both optimizers update the model:

```text
W_new = W_old - learning_rate × gradient
```

Because:

```text
Same starting weights
+
Same synchronized gradients
```

both model copies remain identical.

---

# 4. Simple conceptual implementation

Without DDP:

```python
model = Model()

for batch in dataloader:

    output = model(batch)

    loss = calculate_loss(
        output
    )

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
```

With DDP, the concept is:

```text
GPU 0
    │
    ├── Model Copy
    ├── Data Shard 0
    └── Gradient
            │
            │
            ▼
        ALL REDUCE
            │
            ▼
     Synchronized Gradient
            │
            ▼
       Optimizer Update


GPU 1
    │
    ├── Model Copy
    ├── Data Shard 1
    └── Gradient
```

---

# 5. Basic PyTorch DDP code

Here is a complete minimal example.

## Step 1: Imports

```python
import os

import torch

import torch.nn as nn

import torch.optim as optim

import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP
)

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from torch.utils.data.distributed import (
    DistributedSampler
)
```

---

## Step 2: Initialize distributed training

```python
def setup_ddp():

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

For NVIDIA GPUs:

```text
backend = "nccl"
```

NCCL handles efficient GPU communication.

---

# 6. Create a model

```python
class SimpleModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                10,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)
```

---

# 7. Create a dataset

```python
def create_dataset():

    x = torch.randn(
        10_000,
        10
    )

    y = torch.randn(
        10_000,
        1
    )

    dataset = TensorDataset(
        x,
        y
    )

    return dataset
```

---

# 8. Use DistributedSampler

This is extremely important.

Without `DistributedSampler`, every GPU might process the same data.

```python
dataset = create_dataset()

sampler = DistributedSampler(
    dataset
)
```

With 4 GPUs:

```text
Full Dataset

Example 1
Example 2
Example 3
Example 4
Example 5
Example 6
Example 7
Example 8

        ↓

GPU 0 → 1, 5
GPU 1 → 2, 6
GPU 2 → 3, 7
GPU 3 → 4, 8
```

Create the dataloader:

```python
dataloader = DataLoader(

    dataset,

    batch_size=32,

    sampler=sampler
)
```

Notice:

```python
sampler=sampler
```

You usually should not use:

```python
shuffle=True
```

when using `DistributedSampler`.

The sampler controls shuffling.

---

# 9. Wrap the model with DDP

```python
local_rank = setup_ddp()

model = SimpleModel()

model = model.to(
    local_rank
)
```

Wrap it:

```python
model = DDP(

    model,

    device_ids=[
        local_rank
    ]
)
```

Now PyTorch manages gradient synchronization.

---

# 10. Create optimizer

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

Loss:

```python
loss_function = nn.MSELoss()
```

---

# 11. Complete DDP training loop

```python
def train():

    local_rank = setup_ddp()


    # Model

    model = SimpleModel().to(
        local_rank
    )


    model = DDP(

        model,

        device_ids=[
            local_rank
        ]
    )


    # Dataset

    dataset = create_dataset()


    sampler = DistributedSampler(
        dataset
    )


    dataloader = DataLoader(

        dataset,

        batch_size=32,

        sampler=sampler
    )


    # Optimizer

    optimizer = optim.AdamW(

        model.parameters(),

        lr=1e-3
    )


    loss_function = nn.MSELoss()


    # Training

    for epoch in range(10):

        # Important:
        # Different shuffle order each epoch

        sampler.set_epoch(
            epoch
        )


        for x, y in dataloader:

            x = x.to(
                local_rank
            )

            y = y.to(
                local_rank
            )


            # Forward pass

            prediction = model(
                x
            )


            # Loss

            loss = loss_function(

                prediction,

                y
            )


            # Backward

            optimizer.zero_grad()

            loss.backward()


            # DDP synchronizes gradients
            # automatically during backward()

            optimizer.step()


        if local_rank == 0:

            print(
                f"Epoch {epoch}, "
                f"Loss: {loss.item()}"
            )


    # Clean up

    dist.destroy_process_group()


if __name__ == "__main__":

    train()
```

---

# 12. How do you run it?

Suppose the file is:

```text
train_ddp.py
```

Run with four GPUs:

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_ddp.py
```

This creates:

```text
Process 0 → GPU 0

Process 1 → GPU 1

Process 2 → GPU 2

Process 3 → GPU 3
```

Each process:

```text
Has its own:

Model copy
GPU
Data shard
Optimizer
```

---

# 13. What happens during one training step?

Let's use:

```text
2 GPUs
```

## Step 1: Same model

```text
GPU 0

Model W
```

```text
GPU 1

Model W
```

---

## Step 2: Different batches

```text
GPU 0

Batch A
```

```text
GPU 1

Batch B
```

---

## Step 3: Forward pass

```text
GPU 0

Prediction A
     ↓
Loss A
```

```text
GPU 1

Prediction B
     ↓
Loss B
```

---

## Step 4: Local gradients

```text
GPU 0

Gradient A
```

```text
GPU 1

Gradient B
```

---

## Step 5: All-reduce

```text
GPU 0 gradient ──┐
                 │
                 ▼
             ALL REDUCE
                 │
GPU 1 gradient ──┘
                 │
                 ▼
         Average gradient
```

Now both GPUs have:

```text
Same synchronized gradient
```

---

## Step 6: Update

```text
GPU 0

W_new
```

```text
GPU 1

W_new
```

The models remain synchronized.

---

# 14. Effective batch size

Suppose:

```text
Number of GPUs = 4

Batch size per GPU = 8

Gradient accumulation = 2
```

Effective global batch size:

```text
4 × 8 × 2 = 64
```

Formula:

```python
def effective_batch_size(
    num_gpus,
    batch_per_gpu,
    gradient_accumulation
):

    return (
        num_gpus
        *
        batch_per_gpu
        *
        gradient_accumulation
    )


print(

    effective_batch_size(

        num_gpus=4,

        batch_per_gpu=8,

        gradient_accumulation=2
    )
)
```

Output:

```text
64
```

This is important when tuning:

```text
Learning rate
Warmup
Training steps
```

because increasing the number of GPUs changes the global batch size.

---

# 15. DDP for LLM fine-tuning

The same idea applies to LLMs.

```text
                        Training Dataset
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           GPU 0             GPU 1             GPU 2
              │                │                │
         7B Model          7B Model          7B Model
              │                │                │
         Data shard        Data shard        Data shard
              │                │                │
              ▼                ▼                ▼
            Gradients       Gradients       Gradients
                    │          │          │
                    └──────────┼──────────┘
                               ▼
                         All-Reduce
```

Example using Hugging Face:

```python
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
```

Load model:

```python
model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16
    )
)
```

Training arguments:

```python
training_args = TrainingArguments(

    output_dir="./output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    bf16=True,

    learning_rate=2e-5
)
```

Launch:

```bash
torchrun \
  --nproc_per_node=4 \
  train.py
```

Depending on the training stack, Hugging Face/Accelerate can handle the distributed setup.

---

# 16. Important limitation: DDP does NOT reduce model memory per GPU

This is extremely important.

Suppose:

```text
7B model requires 14 GB
```

With four GPUs using DDP:

```text
GPU 0 → 14 GB model copy

GPU 1 → 14 GB model copy

GPU 2 → 14 GB model copy

GPU 3 → 14 GB model copy
```

Each GPU still stores the complete model.

So DDP helps primarily with:

```text
✓ Faster training
✓ Larger global batch sizes
✓ More training throughput
```

But it does not solve:

```text
✗ Model too large for one GPU
```

For example:

```text
Model requires 40 GB

GPU memory = 24 GB
```

DDP:

```text
GPU 0 → Cannot fit ❌

GPU 1 → Cannot fit ❌

GPU 2 → Cannot fit ❌

GPU 3 → Cannot fit ❌
```

Because every GPU needs a full model replica.

For model memory sharding, you need approaches such as:

```text
FSDP
DeepSpeed ZeRO
Tensor Parallelism
Pipeline Parallelism
```

---

# 17. DDP vs FSDP

| Feature                      | DDP                    | FSDP                       |
| ---------------------------- | ---------------------- | -------------------------- |
| Model replicated on each GPU | Yes                    | No, parameters are sharded |
| Good for                     | Model fits on each GPU | Large model                |
| Data distributed             | Yes                    | Yes                        |
| Gradient synchronization     | Yes                    | Yes                        |
| Memory per GPU               | Higher                 | Lower                      |
| Complexity                   | Lower                  | Higher                     |

Conceptually:

## DDP

```text
GPU 0 → Full Model

GPU 1 → Full Model

GPU 2 → Full Model
```

## FSDP

```text
Full Model

GPU 0 → Part 1

GPU 1 → Part 2

GPU 2 → Part 3
```

---

# 18. DDP vs DataParallel

You might see:

```python
torch.nn.DataParallel
```

and:

```python
torch.nn.parallel.DistributedDataParallel
```

DDP is generally preferred for multi-GPU training.

### DataParallel

```text
One process
     │
     ▼
GPU 0 coordinates work
     │
 ┌───┼────┐
 ▼   ▼    ▼
GPU0 GPU1 GPU2
```

This can create a bottleneck.

### DDP

```text
Process 0 → GPU 0

Process 1 → GPU 1

Process 2 → GPU 2

Process 3 → GPU 3
```

Each process works independently and synchronizes gradients efficiently.

---

# 19. Common DDP mistakes

## Mistake 1: Not using DistributedSampler

Wrong:

```python
DataLoader(
    dataset,
    shuffle=True
)
```

Potentially every process sees overlapping/full data.

Correct:

```python
sampler = DistributedSampler(
    dataset
)

DataLoader(
    dataset,
    sampler=sampler
)
```

---

## Mistake 2: Forgetting `set_epoch`

Wrong:

```python
for epoch in range(epochs):

    for batch in dataloader:
        train(batch)
```

Correct:

```python
for epoch in range(epochs):

    sampler.set_epoch(
        epoch
    )

    for batch in dataloader:

        train(batch)
```

This ensures proper distributed shuffling across epochs.

---

## Mistake 3: Every process saves the model

Wrong:

```python
torch.save(
    model.state_dict(),
    "model.pt"
)
```

Every process might write simultaneously.

Correct:

```python
if local_rank == 0:

    torch.save(
        model.module.state_dict(),
        "model.pt"
    )
```

Because with DDP:

```text
model = DDP(model)
```

the underlying model is:

```text
model.module
```

---

## Mistake 4: Logging from every process

Wrong:

```python
print(loss)
```

You may get:

```text
GPU 0: Loss 1.2
GPU 1: Loss 1.1
GPU 2: Loss 1.3
GPU 3: Loss 1.2
```

Usually log from rank 0:

```python
if local_rank == 0:

    print(loss)
```

---

# 20. Interview-ready answer

> **Distributed Data Parallelism is a technique for training a model across multiple GPUs by keeping a replica of the model on each GPU and splitting the training data across them. Each GPU performs forward and backward propagation on a different mini-batch, producing local gradients. DDP then performs an all-reduce operation to synchronize and usually average the gradients across all processes. Each GPU then applies the same optimizer update, keeping all model replicas synchronized.**
>
> **In PyTorch, I would typically use DistributedDataParallel with one process per GPU, NCCL as the communication backend, and DistributedSampler to ensure each process receives a different shard of the dataset.**
>
> **DDP improves training throughput, but it does not reduce the model memory required per GPU because each GPU still contains a complete model replica. If the model does not fit on a single GPU, I would consider FSDP, DeepSpeed ZeRO, tensor parallelism, or pipeline parallelism.**

## The easiest way to remember

```text
DDP =

Same Model
     ×
Many GPUs

Different Data
     ↓
Local Gradients
     ↓
All-Reduce
     ↓
Same Updated Model
```

For your LLM fine-tuning interview preparation, the most important distinction is:

> **DDP solves training throughput. FSDP and ZeRO primarily help solve per-GPU memory limitations.**
