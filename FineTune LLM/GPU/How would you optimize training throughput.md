# How would you optimize LLM training throughput?

**Training throughput** means how much useful training work you complete per unit time, commonly measured as:

```text
tokens / second
samples / second
steps / second
```

For LLM training, **tokens/second is usually the most useful metric** because samples can have very different sequence lengths.

## Interview answer

> **I optimize training throughput by first profiling where time is spent—data loading, CPU preprocessing, GPU computation, communication, or checkpointing. Then I improve GPU utilization using efficient batching, sequence packing, mixed precision, Flash Attention, optimized kernels, and compilation. I remove data bottlenecks with parallel preprocessing and prefetching. For multi-GPU training, I optimize distributed communication using DDP, FSDP, or DeepSpeed and overlap communication with computation. Finally, I measure tokens/sec, GPU utilization, communication time, and cost per trained token to validate each optimization.**

---

# 1. First measure throughput

Don't optimize blindly.

```python
import time

start = time.time()

trainer.train()

elapsed = time.time() - start

print(f"Training time: {elapsed:.2f} seconds")
```

For a custom loop:

```python
total_tokens = 0
start = time.perf_counter()

for batch in dataloader:
    input_ids = batch["input_ids"]

    total_tokens += input_ids.numel()

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

elapsed = time.perf_counter() - start

tokens_per_second = total_tokens / elapsed

print(f"Tokens/sec: {tokens_per_second:.2f}")
```

For more accurate measurements, avoid including warm-up steps because GPU kernels and compilation can have startup overhead.

---

# 2. Profile before optimizing

Your pipeline is:

```text
Dataset
   ↓
CPU preprocessing
   ↓
DataLoader
   ↓
CPU → GPU transfer
   ↓
Forward pass
   ↓
Backward pass
   ↓
Distributed communication
   ↓
Optimizer step
```

The slowest component determines throughput.

For example:

```text
GPU utilization = 35%
```

might indicate:

```text
Slow DataLoader
Slow tokenization
Slow disk I/O
CPU bottleneck
```

While:

```text
GPU utilization = 95%
```

suggests the GPU is the main bottleneck.

A production approach is to profile with tools such as [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html?utm_source=chatgpt.com) and framework-level metrics.

---

# 3. Increase effective batch size carefully

GPUs are often underutilized with very small batches.

Example:

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1
)
```

If memory allows:

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4
)
```

This can increase:

```text
GPU utilization
+
Tokens/sec
```

But eventually:

```text
Batch size ↑
      ↓
Memory ↑
      ↓
Possible OOM
```

So I benchmark multiple values:

```text
Batch size: 1 → 1,500 tokens/sec
Batch size: 2 → 2,400 tokens/sec
Batch size: 4 → 3,100 tokens/sec
Batch size: 8 → OOM
```

Then choose the best throughput within the memory budget.

---

# 4. Use gradient accumulation correctly

Gradient accumulation helps when the desired **effective batch size** doesn't fit in memory:

```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8
)
```

Effective batch:

```text
2 × 8 = 16
```

However, an important optimization point:

> **Gradient accumulation is mainly for fitting a larger effective batch size. It does not automatically maximize throughput.**

If the GPU can fit a larger micro-batch:

```text
micro batch = 8
```

that may be faster than:

```text
micro batch = 1
accumulation = 8
```

because the latter performs more separate forward/backward passes.

---

# 5. Use BF16 mixed precision

Instead of FP32:

```text
FP32
```

use:

```text
BF16
```

Example:

```python
training_args = TrainingArguments(
    output_dir="./output",
    bf16=True
)
```

Benefits:

```text
Memory ↓
Memory bandwidth ↓
Tensor Core utilization ↑
Throughput ↑
```

For supported modern GPUs, BF16 is generally a strong default.

---

# 6. Use TF32 when appropriate

For some NVIDIA GPUs, TF32 can accelerate matrix multiplication while retaining FP32-oriented workflows.

```python
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

This can improve throughput for eligible operations, but you should validate numerical behavior for your workload.

---

# 7. Use Flash Attention

Standard attention can be expensive.

Conceptually:

```text
Q × Kᵀ
   ↓
Large attention matrix
   ↓
Memory movement
   ↓
Softmax
   ↓
V multiplication
```

Flash Attention uses more efficient GPU memory access and fused kernels.

When supported by the model/runtime, load with an efficient attention implementation, for example:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

Benefits often include:

```text
Attention memory ↓
Attention speed ↑
Long-context throughput ↑
```

Exact support depends on your GPU, model, PyTorch, and Transformers versions.

---

# 8. Reduce padding with dynamic padding

Suppose:

```text
Example 1 = 100 tokens
Example 2 = 200 tokens
Example 3 = 1000 tokens
```

If everything is padded to:

```text
4096 tokens
```

you waste enormous compute.

Instead:

```python
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True
    )
```

Use a data collator for dynamic padding.

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

Now each batch is padded closer to the longest sample **in that batch**.

---

# 9. Bucket examples by length

Dynamic padding can still waste tokens if batches mix:

```text
100 tokens
120 tokens
4000 tokens
```

Instead, group similar-length examples:

```text
Batch 1:
100
120
130
150

Batch 2:
1000
1100
1200
1300
```

Conceptually:

```text
Random batching

100    → padded to 4000
120    → padded to 4000
4000
```

vs:

```text
Length bucketing

100
120
130
```

Less padding means:

```text
Less FLOPs
+
More useful tokens/sec
```

---

# 10. Use sequence packing

For SFT datasets with many short examples:

```text
Example A = 200 tokens
Example B = 300 tokens
Example C = 150 tokens
```

Instead of processing:

```text
[Example A + padding]
[Example B + padding]
[Example C + padding]
```

pack them:

```text
[A | B | C]
```

With TRL:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    packing=True
)
```

This can substantially improve training efficiency for short examples.

You should still ensure the dataset and training objective are compatible with your packing strategy.

---

# 11. Optimize the DataLoader

A slow DataLoader causes:

```text
GPU waiting
     ↓
GPU utilization ↓
     ↓
Tokens/sec ↓
```

Example:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,

    num_workers=8,

    pin_memory=True,

    persistent_workers=True,

    prefetch_factor=4
)
```

### Why?

`num_workers`:

```text
Multiple CPU processes load data
```

`pin_memory=True`:

```text
Faster CPU → GPU transfers
```

`persistent_workers=True`:

```text
Workers stay alive between epochs
```

`prefetch_factor`:

```text
Prepare future batches
while GPU processes current batch
```

Tune these values; more workers are not always faster.

---

# 12. Move data to GPU efficiently

Use non-blocking transfers when the source memory is pinned:

```python
batch = {
    key: value.to(
        "cuda",
        non_blocking=True
    )
    for key, value in batch.items()
}
```

This can allow better overlap between transfer and computation.

---

# 13. Pre-tokenize the dataset

Bad:

```text
Training loop
    ↓
Read text
    ↓
Tokenize
    ↓
GPU
```

Tokenization may become the bottleneck.

Better:

```text
Raw dataset
    ↓
Offline tokenization
    ↓
Save processed dataset
    ↓
Training
```

Example:

```python
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048
    )


tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    num_proc=8
)
```

Then:

```python
tokenized_dataset.save_to_disk(
    "./tokenized_dataset"
)
```

Later:

```python
from datasets import load_from_disk

dataset = load_from_disk(
    "./tokenized_dataset"
)
```

This is especially useful when repeatedly running experiments on the same dataset.

---

# 14. Use optimized optimizers

Optimizer steps can be expensive.

Examples include fused optimizers where supported.

For example:

```python
training_args = TrainingArguments(
    output_dir="./output",
    optim="adamw_torch_fused"
)
```

Another option in some adapter/quantized workflows:

```python
training_args = TrainingArguments(
    output_dir="./output",
    optim="paged_adamw_8bit"
)
```

The best choice depends on your hardware and training configuration.

---

# 15. Use `torch.compile`

For supported workloads:

```python
import torch

model = torch.compile(
    model
)
```

This can optimize parts of the model execution graph.

With Hugging Face:

```python
training_args = TrainingArguments(
    output_dir="./output",
    torch_compile=True
)
```

However, benchmark it.

Sometimes:

```text
First run → slower because compilation

Later runs → faster
```

So don't compare throughput using only the first few steps.

---

# 16. Use gradient checkpointing only when needed

Gradient checkpointing:

```text
Memory ↓
Compute ↑
```

It recomputes some activations during backward propagation.

Therefore:

> **It is primarily a memory optimization, not a throughput optimization.**

If your model fits comfortably without it:

```text
No checkpointing
```

may provide higher throughput.

If checkpointing enables:

```text
Larger micro-batch
```

then overall throughput might improve.

So benchmark both:

```text
Checkpointing ON
vs
Checkpointing OFF
```

---

# 17. Reduce unnecessary sequence length

This can have one of the biggest effects.

Suppose:

```text
Most examples = 800 tokens
```

but:

```python
max_length = 8192
```

is configured.

You're paying for a much larger context window than needed.

Benchmark:

```text
512
1024
2048
4096
```

while tracking:

```text
tokens/sec
quality
loss
```

Choose the smallest context that preserves task quality.

---

# 18. Multi-GPU training

Single GPU:

```text
GPU 0
```

Multi-GPU:

```text
GPU 0 → Batch A
GPU 1 → Batch B
GPU 2 → Batch C
GPU 3 → Batch D
```

This is data parallelism.

Launch with:

```bash
torchrun \
    --nproc_per_node=4 \
    train.py
```

A simplified Accelerate configuration can also manage distributed launches.

---

# 19. Optimize distributed communication

With multiple GPUs:

```text
Forward
   ↓
Backward
   ↓
Gradient synchronization
```

Communication can become the bottleneck.

DeepSpeed can overlap communication and computation:

```json
{
    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

Conceptually:

```text
GPU computes gradients
        │
        ├──── Compute next work
        │
        └──── Communicate gradients
```

If communication and computation overlap, throughput can improve.

---

# 20. Don't always use ZeRO-3

This is important.

```text
ZeRO-3

Memory savings ↑↑↑

Communication ↑
```

If the model fits with:

```text
DDP
or
ZeRO-2
```

those may provide better throughput.

Decision:

```text
Model fits?
   │
   ├── Yes
   │     ↓
   │  DDP / ZeRO-1 / ZeRO-2
   │
   └── No
         ↓
      FSDP / ZeRO-3
```

Choose the **least expensive distributed strategy that fits the model**.

---

# 21. Reduce checkpoint overhead

Saving a huge model can pause training.

Bad:

```text
Save every 10 steps
```

Better:

```python
training_args = TrainingArguments(
    output_dir="./output",

    save_strategy="steps",

    save_steps=1000,

    save_total_limit=3
)
```

You must balance:

```text
Recovery capability
vs
I/O overhead
```

For very large distributed models, checkpoint format and storage bandwidth also matter.

---

# 22. Reduce logging overhead

Bad:

```text
Every step:
    log metrics
    save images
    call external APIs
```

This can slow training.

Better:

```python
TrainingArguments(
    logging_steps=50
)
```

For large distributed jobs:

```text
Rank 0
   ↓
External logging

Other ranks
   ↓
Minimal logging
```

---

# 23. Avoid expensive Python code inside the training loop

Bad:

```python
for batch in dataloader:

    text = tokenizer.decode(
        batch["input_ids"][0]
    )

    print(text)

    output = model(**batch)
```

Decoding and printing can slow training.

Better:

```python
for batch in dataloader:

    output = model(**batch)
```

Keep:

```text
Python overhead ↓
GPU computation ↑
```

---

# 24. Monitor the right metrics

I would monitor:

```text
Training throughput
    ├── Tokens/sec
    ├── Samples/sec
    └── Steps/sec

GPU
    ├── GPU utilization
    ├── GPU memory
    └── SM utilization

Data pipeline
    ├── DataLoader wait time
    └── CPU utilization

Distributed
    ├── All-reduce time
    ├── Communication/computation ratio
    └── GPU imbalance

Training quality
    ├── Training loss
    └── Validation metrics
```

A useful conceptual metric is:

```text
MFU = achieved model FLOPs
      --------------------
      theoretical hardware FLOPs
```

**Model FLOPs Utilization (MFU)** helps indicate how efficiently the accelerator is being used.

---

# 25. A production configuration example

For a large LLM using LoRA:

```python
training_args = TrainingArguments(
    output_dir="./output",

    # Throughput
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    # Precision
    bf16=True,
    tf32=True,

    # Performance
    torch_compile=True,

    # Training
    learning_rate=2e-4,
    num_train_epochs=3,

    # Logging
    logging_steps=20,

    # Checkpointing
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,

    # Data
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    report_to="wandb"
)
```

Then I would benchmark:

```text
Configuration A

Batch = 1
Tokens/sec = 4,000


Configuration B

Batch = 2
Tokens/sec = 6,800


Configuration C

Batch = 4
Tokens/sec = 9,200


Configuration D

Batch = 8
OOM
```

I would choose:

```text
Batch = 4
```

assuming quality is unchanged and GPU utilization is healthy.

---

# 26. Example optimization workflow

My workflow would be:

```text
                 Baseline
                    │
                    ▼
            Measure tokens/sec
                    │
                    ▼
             Profile pipeline
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
    Data slow     GPU slow     Comm slow
       │            │             │
       ▼            ▼             ▼
   Pre-tokenize   BF16         Tune DDP
   More workers   Flash Attn   ZeRO/FSDP
   Prefetch       Compile      Overlap comm
   Packing        Batch tune   Faster network
       │            │             │
       └────────────┼─────────────┘
                    ▼
              Benchmark again
                    │
                    ▼
            Keep improvements
```

---

# Important trade-offs

| Optimization           |      Memory |      Throughput | Trade-off                  |
| ---------------------- | ----------: | --------------: | -------------------------- |
| Larger micro-batch     |      Higher |    Often higher | OOM risk                   |
| Gradient accumulation  |       Lower |    Can be lower | More passes                |
| BF16                   |       Lower |          Higher | Hardware dependent         |
| Flash Attention        |       Lower |          Higher | Compatibility              |
| Dynamic padding        |       Lower |          Higher | Requires good batching     |
| Sequence packing       | Lower waste |          Higher | Dataset handling           |
| Gradient checkpointing |       Lower |     Often lower | Recomputation              |
| LoRA                   |       Lower | Often efficient | Less parameter flexibility |
| ZeRO-3                 |  Much lower |    Can be lower | Communication              |
| `torch.compile`        |     Similar |    Often higher | Compilation overhead       |

---

# Final interview-ready answer

> **To optimize LLM training throughput, I start by measuring tokens per second and profiling the pipeline to identify whether the bottleneck is data loading, GPU computation, distributed communication, or checkpoint I/O. I then improve GPU utilization by tuning the largest feasible micro-batch, using BF16, efficient attention kernels such as Flash Attention, dynamic padding, length bucketing, sequence packing, fused optimizers, and optionally `torch.compile`. I pre-tokenize datasets and tune the DataLoader to avoid GPU starvation. For distributed training, I select the least communication-heavy strategy that fits the model—typically DDP or ZeRO-2 when possible, and FSDP or ZeRO-3 when memory requires parameter sharding. Finally, I benchmark every change using tokens/sec, GPU utilization, communication time, memory, and cost per trained token.**

## The key formula to remember

```text
Training throughput
        =
Useful tokens processed
----------------------------
Training time
```

The goal is:

```text
GPU idle time ↓
Padding waste ↓
Memory bottlenecks ↓
Communication overhead ↓

GPU utilization ↑
Useful tokens/sec ↑
```
