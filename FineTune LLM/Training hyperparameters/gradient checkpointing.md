# What is Gradient Checkpointing?

**Gradient checkpointing** (also called **activation checkpointing**) is a memory-saving technique used during neural network training.

The main idea is:

> Instead of storing the output (activation) of every layer during the forward pass, we store only selected activations called **checkpoints**. During backpropagation, missing activations are recomputed when needed.

This is extremely useful for:

* LLM fine-tuning
* Large transformer models
* Long sequence lengths
* Limited GPU memory
* LoRA/QLoRA training

---

# 1. First understand normal training

Suppose we have a neural network:

```text
Input
  │
  ▼
Layer 1
  │  Activation A1
  ▼
Layer 2
  │  Activation A2
  ▼
Layer 3
  │  Activation A3
  ▼
Layer 4
  │  Activation A4
  ▼
Loss
```

During the **forward pass**, PyTorch calculates activations:

```text
Input → Layer 1 → A1
              ↓
           Layer 2 → A2
              ↓
           Layer 3 → A3
              ↓
           Layer 4 → A4
```

Normally, PyTorch stores:

```text
GPU Memory

A1
A2
A3
A4
```

Why?

Because during backpropagation, the gradients need intermediate activations.

---

# 2. Why are activations needed for backpropagation?

Consider:

[
y = f(x, W)
]

During backpropagation, we need derivatives such as:

[
\frac{\partial L}{\partial W}
]

For many operations, calculating the gradient requires values produced during the forward pass.

For example:

```text
Forward:

x
 ↓
Layer 1
 ↓
A1
 ↓
Layer 2
 ↓
A2
 ↓
Loss
```

During backward:

```text
Loss
 ↓
Need A2
 ↓
Calculate gradient for Layer 2
 ↓
Need A1
 ↓
Calculate gradient for Layer 1
```

Therefore, frameworks normally save activations.

---

# 3. The memory problem with LLMs

Imagine an LLM with 32 transformer layers:

```text
Input
 ↓
Transformer Layer 1
 ↓
Transformer Layer 2
 ↓
Transformer Layer 3
 ↓
...
 ↓
Transformer Layer 32
 ↓
Output
```

Normal training stores activations from many layers:

```text
GPU Memory

Layer 1 activations
Layer 2 activations
Layer 3 activations
...
Layer 32 activations
```

For LLMs, activation memory can be huge.

It depends strongly on:

```text
Batch size
    ×
Sequence length
    ×
Hidden dimension
    ×
Number of layers
```

A simplified intuition:

[
Activation\ Memory
\propto
Batch\ Size
\times
Sequence\ Length
\times
Hidden\ Size
\times
Layers
]

Longer context is especially expensive.

For example:

```text
Batch size = 4

Sequence length:
512 tokens  → lower activation memory
2048 tokens → much higher
8192 tokens → very high
```

---

# 4. How gradient checkpointing solves this

Without gradient checkpointing:

```text
FORWARD PASS

Layer 1 → Store A1
Layer 2 → Store A2
Layer 3 → Store A3
Layer 4 → Store A4
Layer 5 → Store A5
Layer 6 → Store A6
```

With gradient checkpointing:

```text
FORWARD PASS

Layer 1 → Store checkpoint
Layer 2 → Don't store activation
Layer 3 → Don't store activation
Layer 4 → Store checkpoint
Layer 5 → Don't store activation
Layer 6 → Don't store activation
```

GPU memory now looks like:

```text
Without checkpointing:

[A1][A2][A3][A4][A5][A6]

Memory: HIGH
```

With checkpointing:

```text
[Checkpoint A1]      [Checkpoint A4]

Memory: LOWER
```

The missing activations are not permanently stored.

---

# 5. What happens during backpropagation?

Suppose we have:

```text
Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
Layer 6
```

We store checkpoints:

```text
Checkpoint 1 → output of Layer 1
Checkpoint 2 → output of Layer 4
```

During the forward pass:

```text
Input
 ↓
Layer 1
 ↓
Save checkpoint
 ↓
Layer 2
 ↓
Don't save
 ↓
Layer 3
 ↓
Don't save
 ↓
Layer 4
 ↓
Save checkpoint
 ↓
Layer 5
 ↓
Don't save
 ↓
Layer 6
```

Later, during backward:

```text
Need activation from Layer 3
       ↓
Not stored
       ↓
Start from Layer 1 checkpoint
       ↓
Run Layer 2 again
       ↓
Run Layer 3 again
       ↓
Recover required activation
       ↓
Calculate gradient
```

So gradient checkpointing trades:

```text
GPU memory
     ↓

for

Extra computation
```

This is the key idea.

> **Save less memory by recomputing some forward operations during backpropagation.**

---

# 6. Simple analogy

Imagine you are solving:

```text
Step 1
Step 2
Step 3
Step 4
Step 5
```

## Normal approach

You write down every intermediate result:

```text
Notebook:

Step 1 result ✓
Step 2 result ✓
Step 3 result ✓
Step 4 result ✓
Step 5 result ✓
```

Uses more paper (memory).

## Gradient checkpointing

You write down only important checkpoints:

```text
Notebook:

Step 1 ✓
Step 4 ✓
```

When you need Step 3 later:

```text
Start from Step 1
Redo Step 2
Redo Step 3
```

Uses:

```text
Less memory
More computation
```

---

# 7. Basic PyTorch example

PyTorch provides:

```python
torch.utils.checkpoint.checkpoint
```

Consider a model:

```python
import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)

    def forward(self, x):

        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.relu(x)

        x = self.layer4(x)

        return x
```

Normal training:

```python
model = SimpleModel().cuda()

x = torch.randn(
    32,
    1024,
    device="cuda"
)

output = model(x)

loss = output.mean()

loss.backward()
```

The framework stores intermediate tensors needed for backward.

---

# 8. Apply gradient checkpointing manually

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)


    def forward(self, x):

        # Normal forward
        x = self.layer1(x)
        x = torch.relu(x)

        # Checkpoint this layer
        x = checkpoint(
            self.layer2,
            x,
            use_reentrant=False
        )

        x = torch.relu(x)

        # Checkpoint another layer
        x = checkpoint(
            self.layer3,
            x,
            use_reentrant=False
        )

        x = torch.relu(x)

        x = self.layer4(x)

        return x
```

Now:

```python
model = CheckpointModel().cuda()

x = torch.randn(
    32,
    1024,
    device="cuda"
)

output = model(x)

loss = output.mean()

loss.backward()
```

The checkpointed sections can be recomputed during backward instead of retaining all intermediate activations.

---

# 9. Better example: checkpoint a block of layers

Usually, checkpointing individual operations is not the best approach.

Instead, group layers into blocks.

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 4
        )

        self.linear2 = nn.Linear(
            hidden_size * 4,
            hidden_size
        )

        self.activation = nn.GELU()


    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x
```

Now build a larger model:

```python
class LargeModel(nn.Module):

    def __init__(
        self,
        num_layers=12,
        hidden_size=1024
    ):

        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size)
            for _ in range(num_layers)
        ])


    def forward(self, x):

        for layer in self.layers:

            x = checkpoint(
                layer,
                x,
                use_reentrant=False
            )

        return x
```

Each transformer block is checkpointed.

---

# 10. What happens internally?

Suppose:

```text
Input
  │
  ▼
Block 1
  │
  ▼
Block 2
  │
  ▼
Block 3
  │
  ▼
Block 4
  │
  ▼
Loss
```

## Normal training

During forward:

```text
Store:

Input
A1
A2
A3
A4
```

During backward:

```text
Use:

A4
A3
A2
A1
```

No recomputation is needed.

Memory:

```text
████████████████████
```

---

## Gradient checkpointing

During forward:

```text
Store:

Input
Checkpoint 1
Checkpoint 3
```

Do not retain all intermediate activations.

Memory:

```text
████████
```

During backward:

```text
Need A3
 ↓
Recompute Block 2
 ↓
Recompute Block 3
 ↓
Calculate gradients
```

Computation increases:

```text
Forward computation
        +
Partial forward recomputation
```

---

# 11. Memory vs compute trade-off

This is extremely important for interviews.

| Feature                 | Normal Training       | Gradient Checkpointing |
| ----------------------- | --------------------- | ---------------------- |
| Activation memory       | High                  | Lower                  |
| Training speed          | Faster                | Slower                 |
| Forward recomputation   | No                    | Yes                    |
| Larger models possible  | Limited by GPU memory | More possible          |
| Longer context possible | Less                  | More                   |

The trade-off is:

```text
Normal training:

More Memory
Less Compute
```

vs:

```text
Gradient checkpointing:

Less Memory
More Compute
```

---

# 12. Gradient checkpointing with Hugging Face LLMs

For a Hugging Face model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B"
)
```

Enable gradient checkpointing:

```python
model.gradient_checkpointing_enable()
```

For decoder-only causal LMs, it is also commonly necessary to disable the key-value cache during training:

```python
model.config.use_cache = False
```

Complete example:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

model_name = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Disable KV cache during training
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

Why disable cache?

The attention KV cache is primarily useful for **autoregressive inference**.

During training with gradient checkpointing:

```text
KV cache
   +
Stored activations
```

can conflict with the intended memory-saving strategy and is generally not used together in the standard training configuration.

---

# 13. Gradient checkpointing with LoRA

Example:

```python
import torch

from transformers import (
    AutoModelForCausalLM
)

from peft import (
    LoraConfig,
    get_peft_model
)


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",

    torch_dtype=torch.bfloat16
)


# ---------------------------------
# Enable gradient checkpointing
# ---------------------------------

model.gradient_checkpointing_enable()

model.config.use_cache = False


# ---------------------------------
# Configure LoRA
# ---------------------------------

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# ---------------------------------
# Add LoRA adapters
# ---------------------------------

model = get_peft_model(
    model,
    lora_config
)
```

The architecture becomes:

```text
                  LLM

         Frozen Base Model
                 │
                 ▼
         Transformer Blocks
                 │
                 ▼
         Gradient Checkpointing
                 │
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
     Less Memory      More Compute


                 +
                 │
                 ▼

           LoRA Adapters
                 │
                 ▼
            Trainable
```

---

# 14. Gradient checkpointing with QLoRA

This is a very common real-world combination.

```text
QLoRA
  │
  ├── 4-bit base model
  │
  ├── Frozen base weights
  │
  ├── Train LoRA adapters
  │
  └── Gradient checkpointing
```

Code:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)


# ---------------------------------
# 4-bit quantization
# ---------------------------------

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ---------------------------------
# Load quantized model
# ---------------------------------

model = AutoModelForCausalLM.from_pretrained(

    "meta-llama/Llama-3.1-8B",

    quantization_config=bnb_config,

    device_map="auto"
)


# ---------------------------------
# Prepare for QLoRA training
# ---------------------------------

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)


# Disable cache
model.config.use_cache = False


# ---------------------------------
# LoRA configuration
# ---------------------------------

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# ---------------------------------
# Add LoRA adapters
# ---------------------------------

model = get_peft_model(
    model,
    lora_config
)


# Check trainable parameters
model.print_trainable_parameters()
```

This combines several memory-saving techniques:

```text
                     QLoRA Training

                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼

      4-bit NF4        LoRA         Gradient
    Base Model       Adapters      Checkpointing

          │              │              │

    Less weight      Few trainable    Less activation
       memory          parameters       memory
```

This distinction is important:

| Technique              | Saves                                |
| ---------------------- | ------------------------------------ |
| Quantization           | Base model weight memory             |
| LoRA                   | Trainable parameter/optimizer memory |
| Gradient checkpointing | Activation memory                    |
| Mixed precision        | Memory and often compute             |

They solve **different memory problems**.

---

# 15. How much memory does it save?

There is no fixed percentage.

It depends on:

* Number of layers
* Sequence length
* Batch size
* Hidden dimension
* Attention implementation
* Which operations are checkpointed
* GPU and framework

But conceptually:

```text
Without checkpointing:

Activation memory = proportional to number of saved layer activations
```

With checkpointing:

```text
Activation memory = proportional to number of checkpoints
                    + temporary recomputed activations
```

Theoretical memory savings depend on checkpoint strategy. A common strategy can substantially reduce activation memory, while adding noticeable recomputation overhead.

Do not memorize a fixed statement like:

> "Gradient checkpointing always saves exactly 50% memory."

That is not generally true.

---

# 16. Simple memory intuition with code

You can estimate tensor memory like this:

```python
import torch


def tensor_memory_mb(tensor):

    bytes_used = (
        tensor.numel()
        * tensor.element_size()
    )

    return bytes_used / (1024 ** 2)
```

Suppose:

```python
activation = torch.randn(
    4,       # batch size
    2048,    # sequence length
    4096,    # hidden size
    device="cuda",
    dtype=torch.bfloat16
)

print(
    tensor_memory_mb(activation)
)
```

The shape is:

```text
Batch × Sequence × Hidden

4 × 2048 × 4096
```

BF16 uses:

```text
2 bytes/value
```

So one such activation is approximately:

[
4 \times 2048 \times 4096 \times 2
]

which is about **64 MB**.

Now imagine many layers and multiple intermediate tensors inside each transformer block:

```text
Layer 1 activations
Layer 2 activations
Layer 3 activations
...
Layer 32 activations
```

The total can become very large.

Gradient checkpointing avoids retaining many of these tensors across the entire forward pass.

---

# 17. Practical memory measurement in PyTorch

You can measure GPU memory.

```python
import torch


torch.cuda.reset_peak_memory_stats()

output = model(inputs)

loss = output.loss

loss.backward()


peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

print(
    f"Peak GPU memory: "
    f"{peak_memory:.2f} GB"
)
```

Compare without checkpointing:

```python
model.gradient_checkpointing_disable()
```

Then:

```python
# Run a training step
```

And with checkpointing:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False

# Run a training step
```

Then compare:

```text
Without checkpointing:
Peak GPU memory = X GB

With checkpointing:
Peak GPU memory = Y GB
```

You will usually observe:

```text
Y < X
```

while the checkpointed run may take longer.

---

# 18. Gradient checkpointing with a training loop

```python
import torch

model.train()

model.gradient_checkpointing_enable()

model.config.use_cache = False


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


for batch in train_dataloader:

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch[
            "attention_mask"
        ].cuda(),
        labels=batch["labels"].cuda()
    )

    loss = outputs.loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    print(
        "Loss:",
        loss.item()
    )
```

The training loop itself looks almost the same.

The major change is:

```python
model.gradient_checkpointing_enable()
```

Internally, transformer layers use checkpointing logic.

---

# 19. Gradient checkpointing with Hugging Face Trainer

```python
from transformers import (
    TrainingArguments
)


training_args = TrainingArguments(

    output_dir="./output",

    learning_rate=2e-5,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    bf16=True,

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    max_grad_norm=1.0,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05
)
```

The overall configuration might look like:

```text
BF16
  ↓
Reduce memory / efficient compute

Gradient accumulation
  ↓
Increase effective batch size

Gradient checkpointing
  ↓
Reduce activation memory

LoRA
  ↓
Reduce trainable parameters

QLoRA
  ↓
Reduce base-model memory
```

---

# 20. Gradient checkpointing vs gradient accumulation

These are often confused.

## Gradient accumulation

```text
Problem:
Cannot fit a large batch

Solution:
Use multiple small batches
and accumulate gradients
```

Example:

```text
Batch 1 → gradient
Batch 2 → accumulate
Batch 3 → accumulate
Batch 4 → optimizer update
```

It primarily helps simulate a larger effective batch size.

---

## Gradient checkpointing

```text
Problem:
Activations consume too much GPU memory

Solution:
Store fewer activations
and recompute them during backward
```

Comparison:

| Feature                        | Gradient Accumulation  | Gradient Checkpointing   |
| ------------------------------ | ---------------------- | ------------------------ |
| Main purpose                   | Larger effective batch | Reduce activation memory |
| Stores fewer activations       | No                     | Yes                      |
| Recomputes forward operations  | No                     | Yes                      |
| More training steps per update | Yes                    | Yes                      |
| Can be used together           | Yes                    | Yes                      |

---

# 21. Gradient checkpointing vs mixed precision

Again, these solve different problems.

## Mixed precision

```text
Instead of:

FP32 = 4 bytes

Use:

FP16/BF16 = 2 bytes
```

It reduces memory by using smaller data types for eligible computations/tensors.

## Gradient checkpointing

```text
Instead of:

Store every activation

Use:

Store selected activations
Recompute missing activations
```

Comparison:

| Technique              | How memory is saved              |
| ---------------------- | -------------------------------- |
| Mixed precision        | Smaller numerical representation |
| Gradient checkpointing | Store fewer activations          |
| Quantization           | Compress model weights           |
| LoRA                   | Train fewer parameters           |
| Gradient accumulation  | Use smaller micro-batches        |

---

# 22. When should you use gradient checkpointing?

Use it when:

```text
Large model
     +
GPU memory is insufficient
     +
Training/fine-tuning
```

Especially when:

* You get CUDA out-of-memory errors
* Training long-context examples
* Fine-tuning 7B, 13B, 70B models
* Increasing sequence length
* Increasing batch size is impossible because of activation memory

Example:

```text
Without checkpointing:

Llama model
Batch size = 2
Sequence = 4096

CUDA Out Of Memory
```

Try:

```text
Gradient checkpointing
        +
BF16
        +
LoRA / QLoRA
        +
Gradient accumulation
```

---

# 23. Disadvantages

Gradient checkpointing is not free.

## 1. Slower training

Some forward computations are repeated.

```text
Forward pass
     +
Extra recomputation
     +
Backward pass
```

Therefore:

```text
Lower memory
      ↕
More computation
```

## 2. More complex debugging

Some configurations involving:

* custom layers
* distributed training
* caching
* mixed precision

can require additional care.

## 3. Not useful for inference

Gradient checkpointing is primarily a **training technique**.

During inference:

```text
No backward pass
```

Therefore, the activation-storage problem is different, and checkpointing usually provides no benefit.

---

# 24. Strong interview answer

> **Gradient checkpointing, also called activation checkpointing, is a memory optimization technique used during model training. Normally, the framework stores intermediate activations from the forward pass because they are needed for backpropagation. With gradient checkpointing, only selected activations are stored as checkpoints, while the other activations are discarded. During the backward pass, the missing activations are recomputed from the nearest checkpoint. This significantly reduces activation memory at the cost of additional computation and slower training.**

---

# 25. One-line answer

```text
Normal training:

Store activations
→ More memory
→ Faster backward


Gradient checkpointing:

Store fewer activations
→ Less memory
→ Recompute during backward
→ Slower training
```

# Final mental model

For a production LLM fine-tuning setup:

```text
                  Large LLM
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼

      QLoRA         BF16        Checkpointing

   4-bit weights   16-bit ops   Fewer activations
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              Lower GPU memory
                      │
                      ▼
          Fine-tune larger models
                      │
                      ▼
               Trade-off:
             slower training
```

A very good interview sentence is:

> **Quantization saves model-weight memory, LoRA reduces trainable-parameter memory, mixed precision reduces tensor memory and can speed computation, while gradient checkpointing specifically reduces activation memory by recomputing parts of the forward pass during backpropagation.**
