# How would you handle Out-of-Memory (OOM) errors when training an LLM?

In an interview, I would answer:

> **I first identify what is consuming memory—model weights, optimizer states, gradients, activations, sequence length, or memory fragmentation. Then I reduce memory systematically using smaller micro-batches, gradient accumulation, mixed precision, gradient checkpointing, efficient attention, LoRA/QLoRA, and distributed sharding such as FSDP or ZeRO. I monitor GPU memory and avoid blindly reducing batch size without finding the root cause.**

Let's go step by step.

---

# 1. First identify where the OOM happens

OOM can happen during:

```text
Model loading
     ↓
Forward pass
     ↓
Backward pass
     ↓
Optimizer step
     ↓
Generation
```

Each suggests a different problem.

| When OOM happens | Likely cause                            |
| ---------------- | --------------------------------------- |
| Model loading    | Model weights too large                 |
| Forward pass     | Activations / sequence length / batch   |
| Backward pass    | Gradients + activations                 |
| Optimizer step   | Optimizer states                        |
| Generation       | KV cache / too many concurrent requests |

Example:

```python
import torch

print(torch.cuda.memory_summary())
```

Useful metrics:

```python
device = torch.device("cuda")

print(
    "Allocated:",
    torch.cuda.memory_allocated(device) / 1024**3,
    "GB"
)

print(
    "Reserved:",
    torch.cuda.memory_reserved(device) / 1024**3,
    "GB"
)

print(
    "Max allocated:",
    torch.cuda.max_memory_allocated(device) / 1024**3,
    "GB"
)
```

Important distinction:

```text
allocated memory = memory currently used by tensors

reserved memory = memory reserved by PyTorch's caching allocator
```

---

# 2. Solution #1: Reduce micro-batch size

Suppose:

```python
per_device_train_batch_size = 8
```

causes:

```text
CUDA Out of Memory
```

Reduce it:

```python
per_device_train_batch_size = 1
```

But reducing batch size can affect optimization.

So use gradient accumulation.

---

# 3. Solution #2: Gradient accumulation

Instead of:

```text
Batch size = 16
```

which might not fit in GPU memory:

```text
Micro batch = 2
Gradient accumulation = 8
```

Effective batch size:

```text
2 × 8 = 16
```

Code:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    learning_rate=2e-5,

    bf16=True
)
```

Conceptually:

```text
Batch 1 → Forward → Backward
             ↓
          Store gradients

Batch 2 → Forward → Backward
             ↓
          Accumulate

...

Batch 16 → Forward → Backward
             ↓
       Optimizer update
```

This is one of the first things I try.

---

# 4. Solution #3: Use mixed precision

Instead of FP32:

```text
1 parameter = 4 bytes
```

use:

```text
FP16 = 2 bytes
BF16 = 2 bytes
```

For modern GPUs, BF16 is often preferred when supported.

```python
training_args = TrainingArguments(
    output_dir="./output",

    bf16=True
)
```

Or:

```python
training_args = TrainingArguments(
    output_dir="./output",

    fp16=True
)
```

Do not enable both.

---

# 5. Solution #4: Gradient checkpointing

Normally:

```text
Forward pass
      ↓
Store activations
      ↓
Backward pass
```

Storing activations consumes significant GPU memory.

With gradient checkpointing:

```text
Forward pass
      ↓
Store fewer activations
      ↓
Backward pass
      ↓
Recompute required activations
```

Tradeoff:

```text
GPU memory ↓

Training time ↑
```

Enable it:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

With Hugging Face training:

```python
training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True
)
```

For large LLMs, this is extremely useful.

---

# 6. Solution #5: Reduce sequence length

This is often forgotten.

Attention complexity grows significantly as sequence length increases.

For example:

```text
512 tokens
    ↓

2048 tokens
```

can increase activation and attention memory dramatically.

Bad:

```python
MAX_LENGTH = 8192
```

when most training examples are:

```text
300–500 tokens
```

Better:

```python
MAX_LENGTH = 2048
```

Or bucket examples by length.

Example:

```python
MAX_LENGTH = 2048

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )
```

For production training, I would inspect:

```text
P50 token length
P95 token length
P99 token length
```

Then choose a sensible maximum sequence length.

---

# 7. Use dynamic padding

Don't pad every example to the maximum model context length.

Bad:

```python
tokenizer(
    text,
    padding="max_length",
    max_length=4096
)
```

If the actual example has:

```text
200 tokens
```

you waste memory processing:

```text
4096 tokens
```

Better:

```python
tokenizer(
    text,
    truncation=True
)
```

Then dynamically pad within each batch.

Example:

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

This can significantly reduce wasted computation and memory.

---

# 8. Use sequence packing

Suppose your examples are:

```text
Example 1 → 200 tokens
Example 2 → 300 tokens
Example 3 → 150 tokens
```

Without packing:

```text
[200 tokens + padding]
[300 tokens + padding]
[150 tokens + padding]
```

With packing:

```text
[Example1 | Example2 | Example3]
```

This can improve GPU utilization and reduce padding waste.

With TRL:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,

    packing=True
)
```

The exact API depends on your TRL version.

---

# 9. Use LoRA instead of full fine-tuning

Full fine-tuning:

```text
Train all parameters

70B parameters
      ↓
Gradients
      ↓
Optimizer states
```

Very memory expensive.

LoRA:

```text
70B Base Model

Frozen
    +
Small trainable adapters
```

Example:

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

    task_type="CAUSAL_LM"
)
```

This dramatically reduces:

```text
Gradient memory
Optimizer memory
```

---

# 10. Use QLoRA for even more memory reduction

QLoRA:

```text
4-bit Base Model
       +
LoRA adapters
       =
QLoRA
```

Example:

```python
import torch

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Load:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Then:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

Conceptually:

```text
70B FP16 model

≈ 140 GB weights

        ↓

4-bit quantization

≈ 35 GB weights
```

The real footprint is larger due to quantization metadata and runtime overhead, but the reduction is substantial.

---

# 11. Use a memory-efficient optimizer

Adam stores additional state.

Instead of a standard optimizer, consider:

```text
AdamW
```

or memory-efficient variants where appropriate.

Example:

```python
training_args = TrainingArguments(
    output_dir="./output",
    optim="paged_adamw_8bit"
)
```

This is commonly used with QLoRA and bitsandbytes-compatible setups.

---

# 12. Use DeepSpeed ZeRO

For multi-GPU training:

## ZeRO-1

```text
Shard optimizer states
```

## ZeRO-2

```text
Shard optimizer states
+
Gradients
```

## ZeRO-3

```text
Shard optimizer states
+
Gradients
+
Parameters
```

Example:

```json
{
    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 3,

        "contiguous_gradients": true,

        "overlap_comm": true
    }
}
```

Use with:

```python
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="./zero3.json",
    bf16=True
)
```

---

# 13. Use FSDP

PyTorch FSDP shards:

```text
Parameters
Gradients
Optimizer states
```

Conceptually:

```text
GPU 0 → Model shard
GPU 1 → Model shard
GPU 2 → Model shard
GPU 3 → Model shard
```

Example:

```python
training_args = TrainingArguments(
    output_dir="./output",

    fsdp="full_shard auto_wrap",

    fsdp_config={
        "backward_prefetch": "backward_pre",
        "forward_prefetch": "false"
    }
)
```

FSDP is particularly useful for very large models that cannot fit on a single GPU.

---

# 14. CPU offloading

If GPU memory is still insufficient:

```text
GPU
│
├── Active computation
└── Some training state

CPU RAM
│
├── Optimizer state
└── Model state
```

DeepSpeed example:

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

Tradeoff:

```text
GPU memory ↓

Training speed ↓
```

because CPU ↔ GPU transfer is slower.

---

# 15. Clear unused memory correctly

During experimentation:

```python
import gc
import torch

del model

gc.collect()

torch.cuda.empty_cache()
```

Important:

> `torch.cuda.empty_cache()` does **not** free memory held by active tensors.

It only releases unused cached memory back to the allocator/driver.

So this will not fix:

```python
tensor = torch.randn(
    100000,
    100000,
    device="cuda"
)

torch.cuda.empty_cache()
```

because `tensor` still exists.

You need:

```python
del tensor

gc.collect()

torch.cuda.empty_cache()
```

---

# 16. Detect memory leaks

A common training problem:

```python
losses = []

for batch in dataloader:

    loss = model(**batch).loss

    losses.append(loss)
```

This can retain the computation graph.

Bad:

```python
losses.append(loss)
```

Better:

```python
losses.append(
    loss.item()
)
```

Another problem:

```python
outputs = []

for batch in dataloader:

    output = model(**batch)

    outputs.append(output)
```

This can keep GPU tensors alive.

Better:

```python
output = model(**batch)

result = output.logits.detach().cpu()

outputs.append(result)
```

And when no longer needed:

```python
del output
```

---

# 17. Example: handling CUDA OOM dynamically

You can catch OOM errors:

```python
import torch


try:

    outputs = model(
        **batch
    )

    loss = outputs.loss

    loss.backward()


except torch.cuda.OutOfMemoryError:

    print("CUDA OOM detected")

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.empty_cache()

    # reduce batch size / retry logic
```

However, in production training I would **not rely only on catching OOM and retrying**.

I would proactively configure:

```text
Memory budget
+
Batch size
+
Sequence length
+
Checkpointing
+
Distributed strategy
```

Catching OOM is more useful for controlled recovery logic.

---

# 18. Adaptive batch size example

For experimentation:

```python
def train_with_batch_size(
    initial_batch_size
):

    batch_size = initial_batch_size

    while batch_size >= 1:

        try:

            print(
                f"Trying batch size: {batch_size}"
            )

            run_training(
                batch_size=batch_size
            )

            return batch_size


        except torch.cuda.OutOfMemoryError:

            print(
                f"OOM with batch size: {batch_size}"
            )

            torch.cuda.empty_cache()

            batch_size //= 2


    raise RuntimeError(
        "Unable to train even with batch size 1"
    )
```

Example:

```text
Try batch = 16
      ↓
OOM

Try batch = 8
      ↓
OOM

Try batch = 4
      ↓
Success
```

Then use:

```text
Micro batch = 4
Gradient accumulation = 4
```

to preserve:

```text
Effective batch = 16
```

---

# 19. Debugging a real 70B QLoRA OOM

Suppose you are training:

```text
70B Model

4-bit QLoRA

GPU = 80 GB
```

and still get OOM.

I would check in this order:

### Step 1: Reduce micro-batch

```python
per_device_train_batch_size=1
```

---

### Step 2: Reduce sequence length

```text
4096
 ↓
2048
```

If possible.

---

### Step 3: Enable gradient checkpointing

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

---

### Step 4: Verify 4-bit quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)
```

---

### Step 5: Reduce LoRA target modules

Instead of:

```text
Attention
+
MLP
```

try:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

This reduces trainable adapter size.

---

### Step 6: Reduce LoRA rank

```python
r=64
```

↓

```python
r=16
```

LoRA memory and parameter count roughly scale with rank.

---

### Step 7: Use multiple GPUs

Then consider:

```text
FSDP

or

DeepSpeed ZeRO
```

depending on the model and training stack.

---

# 20. OOM during inference is different

For inference, the problem may be:

```text
Model weights
+
KV Cache
+
Concurrent requests
```

For example:

```text
100 concurrent requests

Each request
      ↓
Large context
      ↓
Large KV cache
```

Solutions include:

```text
✓ Limit max tokens
✓ Limit context length
✓ Continuous batching
✓ KV-cache management
✓ Quantization
✓ Request queue
✓ Autoscaling
✓ vLLM / optimized serving
```

Example:

```python
output = model.generate(
    **inputs,

    max_new_tokens=512
)
```

Reducing:

```text
4096 new tokens
```

to:

```text
512 new tokens
```

can substantially reduce generation-time memory pressure.

---

# 21. Production OOM handling architecture

For a production training system:

```text
                 Training Job
                      │
                      ▼
                GPU Monitoring
                      │
                      ▼
               Memory Threshold
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
        Normal                High Memory
                                  │
                                  ▼
                         Reduce Micro Batch
                                  │
                                  ▼
                         Restart From Checkpoint
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
                  Success                    Still OOM
                                                │
                                                ▼
                                    Change Training Strategy
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                            QLoRA            ZeRO-3             FSDP
```

The important production principle is:

> **Checkpoint frequently enough that an OOM recovery does not restart the entire training job.**

Example:

```python
training_args = TrainingArguments(
    output_dir="./checkpoints",

    save_strategy="steps",

    save_steps=500,

    save_total_limit=3
)
```

Resume:

```python
trainer.train(
    resume_from_checkpoint=True
)
```

---

# 22. Best decision order

When I get an OOM:

```text
1. Find where the OOM happens
           ↓
2. Reduce micro-batch size
           ↓
3. Use gradient accumulation
           ↓
4. Enable BF16 / FP16
           ↓
5. Enable gradient checkpointing
           ↓
6. Reduce sequence length
           ↓
7. Use dynamic padding / packing
           ↓
8. Use LoRA
           ↓
9. Use QLoRA
           ↓
10. Use efficient optimizer
           ↓
11. Use ZeRO/FSDP
           ↓
12. CPU/NVMe offloading
```

---

# Interview-ready answer

> **When I encounter an out-of-memory error while fine-tuning an LLM, I first identify whether the failure occurs during model loading, forward pass, backward pass, optimizer update, or inference. Then I reduce memory systematically. I usually start by reducing the micro-batch size and compensating with gradient accumulation, then enable BF16, gradient checkpointing, dynamic padding, and reduce unnecessary sequence length. For large models, I use LoRA or QLoRA so that only adapters are trained. If a single GPU is insufficient, I use distributed sharding with FSDP or DeepSpeed ZeRO. Finally, I consider CPU offloading if necessary. I also monitor allocated versus reserved GPU memory and check for memory leaks caused by retaining computation graphs or GPU tensors.**

## Best one-line answer

```text
OOM is solved by reducing:

Batch memory
+
Activation memory
+
Model memory
+
Optimizer memory
```

using:

```text
Gradient Accumulation
+
BF16
+
Checkpointing
+
Shorter Sequences
+
LoRA/QLoRA
+
ZeRO/FSDP
```

That is the systematic approach you should explain in an interview.
