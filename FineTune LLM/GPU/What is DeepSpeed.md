# What is DeepSpeed?

**DeepSpeed** is an open-source deep learning optimization library designed to make **large-scale model training and inference** more efficient. It works primarily with PyTorch and provides features such as distributed training, mixed precision, gradient accumulation, model parallelism, pipeline parallelism, and—most importantly—**ZeRO (Zero Redundancy Optimizer)**. ([DeepSpeed][1])

You can think of it like this:

```text
PyTorch
   +
Distributed Training
   +
Memory Optimization
   +
Mixed Precision
   +
ZeRO
   +
CPU/NVMe Offloading
   =
DeepSpeed
```

---

# 1. Why do we need DeepSpeed?

Suppose you want to train a 7B or 70B LLM.

Normal training requires memory for:

```text
Model Parameters
      +
Gradients
      +
Optimizer States
      +
Activations
```

For example:

```text
7B Model
   │
   ├── Model weights
   ├── Gradients
   ├── Adam optimizer states
   └── Activations
```

This can easily exceed the memory available on a GPU.

DeepSpeed helps by:

```text
✓ Distributing work across GPUs
✓ Sharding optimizer states
✓ Sharding gradients
✓ Sharding model parameters
✓ Using mixed precision
✓ Offloading data to CPU
✓ Offloading data to NVMe
✓ Supporting pipeline/model parallelism
```

Its core technology is **ZeRO**.

---

# 2. The most important DeepSpeed concept: ZeRO

**ZeRO = Zero Redundancy Optimizer**

Normally, with DDP:

```text
GPU 0                    GPU 1

Full Model               Full Model
Full Gradients           Full Gradients
Full Optimizer           Full Optimizer
```

There is a lot of duplicated memory.

DeepSpeed tries to eliminate that redundancy.

---

# 3. ZeRO Stage 1

ZeRO Stage 1 shards **optimizer states**.

## Normal DDP

```text
GPU 0

Model              FULL
Gradients          FULL
Optimizer          FULL
```

```text
GPU 1

Model              FULL
Gradients          FULL
Optimizer          FULL
```

## ZeRO Stage 1

```text
GPU 0

Model              FULL
Gradients          FULL
Optimizer          PART 1
```

```text
GPU 1

Model              FULL
Gradients          FULL
Optimizer          PART 2
```

The optimizer state is distributed across GPUs. ([GitHub][2])

Configuration:

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

---

# 4. ZeRO Stage 2

ZeRO Stage 2 shards:

```text
✓ Optimizer states
✓ Gradients
```

Conceptually:

```text
GPU 0

Model              FULL
Gradients          PART 1
Optimizer          PART 1
```

```text
GPU 1

Model              FULL
Gradients          PART 2
Optimizer          PART 2
```

Configuration:

```json
{
  "zero_optimization": {
    "stage": 2
  }
}
```

DeepSpeed uses partitioning so each process retains only the gradient state associated with its partition. ([DeepSpeed][3])

---

# 5. ZeRO Stage 3

ZeRO Stage 3 shards:

```text
✓ Optimizer states
✓ Gradients
✓ Model parameters
```

This provides the largest memory savings.

```text
GPU 0

Model              PART 1
Gradients          PART 1
Optimizer          PART 1
```

```text
GPU 1

Model              PART 2
Gradients          PART 2
Optimizer          PART 2
```

```text
GPU 2

Model              PART 3
Gradients          PART 3
Optimizer          PART 3
```

Configuration:

```json
{
  "zero_optimization": {
    "stage": 3
  }
}
```

During training, ZeRO-3 gathers the parameters needed for computation and partitions them again afterward. ([DeepSpeed][3])

---

# 6. ZeRO stages comparison

| Feature                  | DDP        | ZeRO-1     | ZeRO-2     | ZeRO-3  |
| ------------------------ | ---------- | ---------- | ---------- | ------- |
| Model parameters         | Replicated | Replicated | Replicated | Sharded |
| Gradients                | Replicated | Replicated | Sharded    | Sharded |
| Optimizer states         | Replicated | Sharded    | Sharded    | Sharded |
| Memory saving            | Low        | Good       | Better     | Highest |
| Communication complexity | Low        | Medium     | Medium     | Higher  |

The easiest way to remember:

```text
ZeRO-1
Optimizer states

ZeRO-2
Optimizer states
+
Gradients

ZeRO-3
Optimizer states
+
Gradients
+
Parameters
```

---

# 7. DeepSpeed vs DDP vs FSDP

You just learned DDP and FSDP, so this comparison is important.

## DDP

```text
GPU 0 → Full Model
GPU 1 → Full Model
GPU 2 → Full Model
```

Best when:

> The model already fits on each GPU and you mainly want higher training throughput.

---

## FSDP

```text
GPU 0 → Model shard
GPU 1 → Model shard
GPU 2 → Model shard
```

PyTorch's native approach for sharding model states.

---

## DeepSpeed ZeRO

```text
GPU 0 → State shard
GPU 1 → State shard
GPU 2 → State shard
```

DeepSpeed provides:

```text
ZeRO
+
Offloading
+
Pipeline parallelism
+
Tensor/model parallelism
+
Training/inference optimizations
```

A useful interview comparison:

| Technology | Main Purpose                                         |
| ---------- | ---------------------------------------------------- |
| DDP        | Faster distributed training                          |
| FSDP       | PyTorch-native memory sharding                       |
| DeepSpeed  | Large-model training/inference optimization platform |
| ZeRO       | DeepSpeed's memory-sharding technology               |

---

# 8. Install DeepSpeed

```bash
pip install deepspeed
```

DeepSpeed's official getting-started documentation describes it as a PyTorch-compatible engine that wraps models and handles distributed setup, optimization, training, and checkpointing. ([DeepSpeed][1])

[DeepSpeed getting started guide](https://www.deepspeed.ai/getting-started/?utm_source=chatgpt.com)

---

# 9. Basic DeepSpeed training example

Let's start with a normal PyTorch model.

```python
import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)
```

---

## DeepSpeed configuration

Create:

```text
ds_config.json
```

```json
{
    "train_batch_size": 32,

    "fp16": {
        "enabled": true
    },

    "optimizer": {
        "type": "AdamW",

        "params": {
            "lr": 0.0001
        }
    },

    "zero_optimization": {
        "stage": 2
    }
}
```

This enables:

```text
Mixed precision
+
AdamW
+
ZeRO Stage 2
```

---

# 10. Initialize DeepSpeed

```python
import deepspeed


model = SimpleModel()

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

The returned `model_engine` replaces much of the normal training management. DeepSpeed's training engine exposes the core forward, backward, and step APIs. ([DeepSpeed][1])

---

# 11. Normal PyTorch vs DeepSpeed

## Normal PyTorch

```python
optimizer.zero_grad()

output = model(x)

loss = loss_function(
    output,
    y
)

loss.backward()

optimizer.step()
```

---

## DeepSpeed

```python
output = model_engine(x)

loss = loss_function(
    output,
    y
)

model_engine.backward(
    loss
)

model_engine.step()
```

The important methods are:

```text
model_engine()
        ↓
Forward

model_engine.backward()
        ↓
Backward

model_engine.step()
        ↓
Optimizer update
```

DeepSpeed handles distributed and mixed-precision behavior according to its configuration. ([DeepSpeed][1])

---

# 12. Complete DeepSpeed training example

```python
import torch
import torch.nn as nn
import deepspeed


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)


def train():

    model = SimpleModel()

    loss_function = nn.MSELoss()


    model_engine, optimizer, _, _ = (
        deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config="ds_config.json"
        )
    )


    for epoch in range(10):

        for step in range(100):

            x = torch.randn(
                32,
                100,
                device=model_engine.device
            )

            y = torch.randn(
                32,
                10,
                device=model_engine.device
            )


            # Forward

            predictions = model_engine(
                x
            )


            # Calculate loss

            loss = loss_function(
                predictions,
                y
            )


            # Backward

            model_engine.backward(
                loss
            )


            # Update

            model_engine.step()


        if model_engine.global_rank == 0:

            print(
                f"Epoch: {epoch}, "
                f"Loss: {loss.item():.4f}"
            )
```

---

# 13. Run DeepSpeed on multiple GPUs

Suppose:

```text
train.py
```

Run:

```bash
deepspeed \
    --num_gpus=4 \
    train.py
```

Conceptually:

```text
DeepSpeed Launcher

        │
        ▼

Process 0 → GPU 0

Process 1 → GPU 1

Process 2 → GPU 2

Process 3 → GPU 3
```

DeepSpeed can also launch multi-node jobs using hostfiles or other supported distributed environment setups. ([DeepSpeed][1])

---

# 14. DeepSpeed with Hugging Face

This is one of the most common real-world uses.

Create:

```text
ds_config.json
```

```json
{
    "fp16": {
        "enabled": "auto"
    },

    "bf16": {
        "enabled": "auto"
    },

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
    },

    "gradient_accumulation_steps": "auto",

    "train_micro_batch_size_per_gpu": "auto"
}
```

Then use Hugging Face training arguments:

```python
from transformers import TrainingArguments


training_args = TrainingArguments(

    output_dir="./output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-5,

    bf16=True,

    deepspeed="./ds_config.json"
)
```

Now `Trainer` can use DeepSpeed:

```python
from transformers import Trainer


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset
)


trainer.train()
```

This is convenient because you don't need to manually write:

```python
deepspeed.initialize(...)
```

for every training project.

---

# 15. DeepSpeed with a 7B LLM

Example:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)


MODEL_NAME = "meta-llama/Llama-3.1-8B"
```

Load:

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)
```

Training:

```python
training_args = TrainingArguments(

    output_dir="./llama-output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    learning_rate=2e-5,

    bf16=True,

    gradient_checkpointing=True,

    deepspeed="./ds_config.json"
)
```

Then:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset
)

trainer.train()
```

A typical production setup may combine:

```text
DeepSpeed
+
ZeRO-3
+
BF16
+
Gradient Checkpointing
+
Gradient Accumulation
```

---

# 16. CPU Offloading

Suppose GPU memory is insufficient.

DeepSpeed can move some training state to CPU.

```text
GPU

Current computation
Current activations
Some parameters
```

```text
CPU

Optimizer states
Some parameters
```

Example:

```json
{
    "zero_optimization": {

        "stage": 3,

        "offload_optimizer": {
            "device": "cpu"
        },

        "offload_param": {
            "device": "cpu"
        }
    }
}
```

This can significantly reduce GPU memory pressure, but introduces CPU↔GPU transfer overhead. DeepSpeed's ZeRO documentation also describes ZeRO-Infinity as extending offloading to CPU and NVMe for very large models. ([DeepSpeed][3])

---

# 17. ZeRO-Offload

Conceptually:

```text
Without offloading

GPU

Model
Gradients
Optimizer
```

With offloading:

```text
GPU

Model computation
Gradients


CPU

Optimizer states
Parameters (optional)
```

Memory usage:

```text
GPU memory ↓↓↓

CPU RAM usage ↑↑
```

Trade-off:

```text
Memory improves

but

CPU ↔ GPU communication can slow training
```

---

# 18. ZeRO-Infinity

For extremely large models:

```text
GPU
 │
 ▼
CPU RAM
 │
 ▼
NVMe Storage
```

Conceptually:

```text
Fastest
GPU
 │
 │ Offload
 ▼
CPU RAM
 │
 │ Offload
 ▼
NVMe
Slowest
```

DeepSpeed can use this hierarchy to extend available memory beyond GPU VRAM. ([DeepSpeed][3])

---

# 19. DeepSpeed supports multiple parallelism strategies

DeepSpeed is not just ZeRO.

It can support combinations of:

```text
Data Parallelism
       +
Tensor Parallelism
       +
Pipeline Parallelism
```

This is sometimes called:

```text
3D Parallelism
```

Conceptually:

```text
                    Large Model

                         │

            ┌────────────┼────────────┐

            ▼            ▼            ▼

        Data          Tensor        Pipeline
      Parallel        Parallel      Parallel
```

### Data parallelism

```text
Same model
Different data
```

### Tensor parallelism

```text
Split one large layer
across multiple GPUs
```

### Pipeline parallelism

```text
GPU 0 → Layers 1–10

GPU 1 → Layers 11–20

GPU 2 → Layers 21–30
```

DeepSpeed documents support for pipeline parallelism and combinations of data, model/tensor, and pipeline parallel techniques. ([DeepSpeed][4])

---

# 20. DeepSpeed vs FSDP

Since you just asked about FSDP, this is important.

## FSDP

```text
PyTorch Native
```

Provides:

```text
✓ Parameter sharding
✓ Gradient sharding
✓ Optimizer state sharding
✓ Distributed training
```

---

## DeepSpeed

Provides:

```text
✓ ZeRO-1
✓ ZeRO-2
✓ ZeRO-3
✓ CPU Offloading
✓ NVMe Offloading
✓ Mixed Precision
✓ Pipeline Parallelism
✓ Tensor Parallelism
✓ Large-model training optimizations
✓ Training/inference tooling
```

---

## Comparison

| Feature              | FSDP                          | DeepSpeed                  |
| -------------------- | ----------------------------- | -------------------------- |
| Part of PyTorch      | Yes                           | No, external library       |
| Parameter sharding   | Yes                           | ZeRO-3                     |
| Gradient sharding    | Yes                           | ZeRO-2/3                   |
| Optimizer sharding   | Yes                           | ZeRO-1/2/3                 |
| CPU offload          | Yes, supported configurations | Yes                        |
| NVMe offload         | Not its primary focus         | Yes via ZeRO-Infinity      |
| Pipeline parallelism | Requires additional setup     | Built-in ecosystem support |
| Tensor parallelism   | Separate approach             | Supported                  |
| LLM ecosystem        | Strong                        | Strong                     |
| Configuration        | Python APIs                   | Often JSON config + APIs   |

A practical summary:

```text
FSDP
=
PyTorch-native sharding

DeepSpeed
=
Large-scale training optimization framework
```

---

# 21. When would I use DeepSpeed?

### Use DDP when:

```text
✓ Model fits on every GPU
✓ Need more throughput
✓ Simpler distributed training
```

### Use FSDP when:

```text
✓ Model doesn't fit on one GPU
✓ Prefer PyTorch-native tooling
✓ Need model-state sharding
```

### Use DeepSpeed when:

```text
✓ Very large LLM
✓ Need ZeRO
✓ Need CPU/NVMe offloading
✓ Need complex distributed optimization
✓ Need combinations of parallelism strategies
```

---

# Interview-ready answer

> **DeepSpeed is a deep learning optimization framework built primarily for efficient distributed training and inference of large models. Its most important feature is ZeRO, or Zero Redundancy Optimizer, which removes memory duplication across data-parallel GPUs.**
>
> **ZeRO Stage 1 shards optimizer states, Stage 2 shards optimizer states and gradients, and Stage 3 additionally shards model parameters. DeepSpeed can also use mixed precision, gradient accumulation, CPU/NVMe offloading, tensor parallelism, and pipeline parallelism.**
>
> **In a real LLM fine-tuning project, I would typically use DeepSpeed ZeRO-2 or ZeRO-3 with BF16, gradient checkpointing, and gradient accumulation. If GPU memory is extremely constrained, I could offload optimizer states or parameters to CPU, while understanding that offloading may reduce training speed because of data transfer overhead.** ([DeepSpeed][1])

## Easiest way to remember

```text
DeepSpeed
    │
    ├── Distributed Training
    │
    ├── Mixed Precision
    │
    ├── ZeRO
    │     ├── Stage 1 → Optimizer
    │     ├── Stage 2 → Optimizer + Gradients
    │     └── Stage 3 → Optimizer + Gradients + Parameters
    │
    ├── CPU Offload
    │
    ├── NVMe Offload
    │
    ├── Tensor Parallelism
    │
    └── Pipeline Parallelism
```

**One-line answer:**

> **DeepSpeed is a large-model training and inference optimization framework, and ZeRO is its key technology for reducing redundant GPU memory usage.**

[1]: https://www.deepspeed.ai/getting-started/?utm_source=chatgpt.com "Getting Started - DeepSpeed"
[2]: https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/zero.md?utm_source=chatgpt.com "DeepSpeed/docs/_tutorials/zero.md at master · deepspeedai/DeepSpeed · GitHub"
[3]: https://www.deepspeed.ai/tutorials/zero/?utm_source=chatgpt.com "Zero Redundancy Optimizer - DeepSpeed"
[4]: https://www.deepspeed.ai/training/?utm_source=chatgpt.com "Training Overview and Features - DeepSpeed"
