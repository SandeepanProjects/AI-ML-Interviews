# How would you fine-tune a 70B model with limited GPU resources?

For a **70B model**, the answer is very different from fine-tuning a 7B model.

The main principle is:

> **With limited GPU resources, I would avoid full fine-tuning and use parameter-efficient fine-tuning—primarily QLoRA or LoRA with distributed sharding. I would quantize and freeze the 70B base model, train only adapters, reduce activation memory with gradient checkpointing, and use small micro-batches with gradient accumulation.**

---

# 1. Why is a 70B model difficult to fine-tune?

A 70B model has approximately:

```text
70 billion parameters
```

## Model weights only

### FP32

```text
70B × 4 bytes
≈ 280 GB
```

### FP16/BF16

```text
70B × 2 bytes
≈ 140 GB
```

### 8-bit

```text
70B × 1 byte
≈ 70 GB
```

### 4-bit

```text
70B × 0.5 bytes
≈ 35 GB
```

These are simplified raw-weight estimates.

Actual GPU memory usage is higher because of:

```text
Quantization metadata
CUDA runtime
Temporary buffers
Activations
KV/cache-related allocations
LoRA parameters
Optimizer states
```

So on:

```text
1 × 24GB GPU
```

you generally cannot load a 70B model entirely in FP16.

Even standard 4-bit loading may be too tight for training because you still need memory for computation and activations.

---

# 2. The strategy depends on your available GPUs

There are several scenarios.

## Scenario A: One 24GB GPU

```text
24 GB GPU
```

Possible:

```text
⚠️ Very constrained
```

Usually requires:

```text
4-bit quantization
+
QLoRA
+
CPU/RAM offloading
+
Small sequence length
+
Batch size = 1
```

But it will often be slow and can be difficult to make stable.

---

## Scenario B: Two to four GPUs

For example:

```text
4 × 24GB GPUs
```

Total memory:

```text
96 GB
```

You can use:

```text
Distributed sharding
+
QLoRA
+
FSDP
or
DeepSpeed ZeRO
```

---

## Scenario C: 8 × 80GB GPUs

```text
8 × 80GB
```

You can realistically consider:

```text
LoRA
QLoRA
Full fine-tuning
FSDP
ZeRO-3
```

depending on context length and optimizer.

---

# 3. Best strategy with limited resources: QLoRA

The architecture:

```text
                    70B Base Model
                          │
                          ▼
                    4-bit NF4
                          │
                          ▼
                  Base weights frozen
                          │
                          ▼
                   Train LoRA layers
                          │
                          ▼
                   Small optimizer
                          │
                          ▼
                    Updated Adapter
```

Instead of training:

```text
70,000,000,000 parameters
```

you train only:

```text
A small percentage
```

The number depends on:

```text
LoRA rank
Target layers
Architecture
```

---

# 4. QLoRA memory concept

Without QLoRA:

```text
70B FP16 weights
≈ 140 GB
```

With QLoRA:

```text
70B model
      ↓
4-bit storage
      ↓
~35 GB theoretical raw storage
```

Then add:

```text
Quantization overhead
+
LoRA adapters
+
Optimizer states
+
Activations
+
CUDA overhead
```

This is why a single 24GB GPU remains difficult.

With multiple GPUs:

```text
GPU 0 → Model shard
GPU 1 → Model shard
GPU 2 → Model shard
GPU 3 → Model shard
```

the memory can be distributed.

---

# 5. Architecture I would recommend

For limited multi-GPU resources:

```text
                    Training Dataset
                           │
                           ▼
                    Tokenization
                           │
                           ▼
                  Sequence Packing
                           │
                           ▼
                  70B Base Model
                           │
                           ▼
                    4-bit NF4
                           │
                           ▼
                Distributed Sharding
                 ┌────────┼────────┐
                 ▼        ▼        ▼
               GPU 0    GPU 1    GPU N
                 │        │        │
                 └────────┼────────┘
                           │
                           ▼
                     LoRA Adapters
                           │
                           ▼
                      BF16 Compute
                           │
                           ▼
                Gradient Checkpointing
                           │
                           ▼
                     Save Adapter
```

---

# 6. Install the stack

```bash
pip install -U \
    torch \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes
```

For distributed training, you may also use:

```bash
pip install deepspeed
```

or use PyTorch FSDP.

---

# 7. Dataset example

```json
{"instruction":"Explain RAG.","input":"","output":"RAG retrieves relevant documents before generating a response."}
{"instruction":"Explain LoRA.","input":"","output":"LoRA trains low-rank adapter matrices while freezing the base model."}
```

Save as:

```text
data/train.jsonl
```

---

# 8. Format the dataset

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)


def format_example(example):

    instruction = example["instruction"]
    user_input = example.get("input", "")
    output = example["output"]

    if user_input:
        text = f"""### Instruction:
{instruction}

### Input:
{user_input}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}


dataset = dataset.map(format_example)
```

---

# 9. Configure 4-bit QLoRA

```python
import torch

from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(

    # Quantize base model
    load_in_4bit=True,

    # Recommended QLoRA quantization type
    bnb_4bit_quant_type="nf4",

    # Reduces quantization overhead
    bnb_4bit_use_double_quant=True,

    # Compute in BF16
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Conceptually:

```text
FP16 weights
   ↓
14 GB for 7B

70B
   ↓
140 GB
```

QLoRA:

```text
70B
   ↓
4-bit
   ↓
~35 GB raw weight equivalent
```

---

# 10. Load the 70B model

```python
from transformers import (
    AutoModelForCausalLM
)


MODEL_NAME = "your-70b-model"


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto",

    low_cpu_mem_usage=True
)
```

`device_map="auto"` may distribute the model across visible GPUs.

However, an important production point:

> For serious distributed training, don't blindly rely on `device_map="auto"` as your training strategy. Use an explicit distributed framework such as Accelerate + FSDP or DeepSpeed, and validate that the model placement is compatible with the quantization and PEFT setup.

---

# 11. CPU offloading if GPU memory is limited

You can conceptually split:

```text
GPU
│
├── Active layers
│
CPU RAM
│
├── Offloaded layers
```

Example:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    max_memory={
        0: "22GiB",
        "cpu": "128GiB"
    },

    offload_folder="./offload"
)
```

This can allow a large model to run with limited GPU memory.

But:

```text
GPU ↔ CPU transfer
        ↓
Training throughput decreases
```

This is generally a last resort.

---

# 12. Prepare the model for k-bit training

```python
from peft import prepare_model_for_kbit_training


model = prepare_model_for_kbit_training(
    model
)
```

Enable gradient checkpointing:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

This is particularly important for 70B training.

---

# 13. Configure LoRA

For a constrained setup, start with attention projections:

```python
from peft import LoraConfig


lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

Why start with fewer modules?

```text
Fewer target layers
        ↓
Fewer LoRA parameters
        ↓
Less optimizer memory
```

If quality is insufficient:

```text
q_proj
v_proj
```

expand to:

```text
q_proj
k_proj
v_proj
o_proj
```

or other architecture-specific linear layers.

---

# 14. Apply LoRA

```python
from peft import get_peft_model


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()
```

You should see something conceptually like:

```text
Trainable parameters: tens of millions
Total parameters: 70 billion
```

The base model remains frozen.

---

# 15. Configure training for limited GPU resources

A conservative starting configuration:

```python
from transformers import TrainingArguments


training_args = TrainingArguments(

    output_dir="./output",

    # Very small micro-batch
    per_device_train_batch_size=1,

    # Effective batch
    gradient_accumulation_steps=32,

    # Precision
    bf16=True,

    # Training
    learning_rate=2e-4,

    num_train_epochs=3,

    # Memory-efficient optimizer
    optim="paged_adamw_8bit",

    # Logging
    logging_steps=10,

    # Checkpoint
    save_strategy="steps",
    save_steps=500,

    save_total_limit=2,

    # Data
    dataloader_num_workers=4,

    report_to="none"
)
```

Effective batch size:

```text
Micro Batch
×
Gradient Accumulation
×
Number of GPUs
```

For 4 GPUs:

```text
1 × 32 × 4 = 128
```

---

# 16. Use a reasonable sequence length

Do not start with:

```text
8192
```

unless your use case truly requires it.

Start with:

```python
MAX_LENGTH = 1024
```

or:

```python
MAX_LENGTH = 2048
```

Why?

Activations are a major source of training memory.

```text
Sequence length ↑
       ↓
Activation memory ↑
       ↓
Attention compute ↑
```

A good optimization strategy:

```text
Start: 1024
   ↓
Measure quality
   ↓
Try: 2048
   ↓
Measure VRAM
   ↓
Try longer only if needed
```

---

# 17. Full training example

Below is a simplified QLoRA training script for a 70B model.

```python
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import SFTTrainer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "your-70b-model"

OUTPUT_DIR = "./70b-output"

MAX_SEQ_LENGTH = 1024


# ============================================================
# Load Dataset
# ============================================================

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)


def format_example(example):

    instruction = example["instruction"]
    user_input = example.get("input", "")
    output = example["output"]

    if user_input:

        text = f"""### Instruction:
{instruction}

### Input:
{user_input}

### Response:
{output}"""

    else:

        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}


dataset = dataset.map(
    format_example
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


tokenizer.padding_side = "right"


# ============================================================
# 4-bit QLoRA
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)


# ============================================================
# Load Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto",

    low_cpu_mem_usage=True
)


# ============================================================
# Prepare for QLoRA
# ============================================================

model = prepare_model_for_kbit_training(
    model
)


model.gradient_checkpointing_enable()

model.config.use_cache = False


# ============================================================
# LoRA
# ============================================================

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

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


model.print_trainable_parameters()


# ============================================================
# Training
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=32,

    num_train_epochs=3,

    learning_rate=2e-4,

    bf16=True,

    optim="paged_adamw_8bit",

    logging_steps=10,

    save_strategy="steps",

    save_steps=500,

    save_total_limit=2,

    report_to="none"
)


# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    args=training_args,

    dataset_text_field="text",

    tokenizer=tokenizer,

    max_seq_length=MAX_SEQ_LENGTH
)


# ============================================================
# Train
# ============================================================

trainer.train()


# ============================================================
# Save LoRA adapter
# ============================================================

trainer.save_model(
    "./70b-lora-adapter"
)


tokenizer.save_pretrained(
    "./70b-lora-adapter"
)
```

---

# 18. Multi-GPU approach using Accelerate

For multiple GPUs, configure:

```bash
accelerate config
```

Then launch:

```bash
accelerate launch \
    --num_processes=4 \
    train.py
```

Conceptually:

```text
Process 0 → GPU 0
Process 1 → GPU 1
Process 2 → GPU 2
Process 3 → GPU 3
```

For a large model, the exact strategy should be configured through Accelerate, DeepSpeed, or FSDP rather than assuming ordinary data parallelism will fit the entire model on every GPU.

---

# 19. DeepSpeed ZeRO-3 configuration

For a model that cannot fit fully on each GPU, ZeRO-3 can shard:

```text
Parameters
+
Gradients
+
Optimizer states
```

Example:

```json
{
    "bf16": {
        "enabled": true
    },

    "zero_optimization": {

        "stage": 3,

        "overlap_comm": true,

        "contiguous_gradients": true,

        "reduce_scatter": true
    }
}
```

Then:

```python
training_args = TrainingArguments(

    output_dir="./output",

    deepspeed="./zero3.json",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    bf16=True
)
```

Conceptually:

```text
Without sharding:

GPU 0 → Entire model
GPU 1 → Entire model
GPU 2 → Entire model
GPU 3 → Entire model
```

With ZeRO-3:

```text
GPU 0 → Model shard
GPU 1 → Model shard
GPU 2 → Model shard
GPU 3 → Model shard
```

---

# 20. FSDP alternative

FSDP also shards:

```text
Model parameters
Gradients
Optimizer states
```

Conceptually:

```text
              70B Model

        ┌────────┼────────┐
        ▼        ▼        ▼

      GPU 0    GPU 1    GPU 2
       1/3      1/3      1/3
```

In practice, you would configure FSDP through your distributed training stack and benchmark against ZeRO because the best choice depends on model architecture, hardware, interconnect, and software versions.

---

# 21. Sequence packing

A 70B model is expensive to train.

You should avoid wasting tokens.

Suppose:

```text
Example 1 = 200 tokens
Example 2 = 300 tokens
Example 3 = 150 tokens
```

Without packing:

```text
[200 + padding]
[300 + padding]
[150 + padding]
```

With packing:

```text
[Example 1 | Example 2 | Example 3]
```

This increases:

```text
Useful tokens/GPU second
```

which is the throughput metric that matters.

---

# 22. Use CPU offloading only when necessary

For extremely constrained environments:

```text
GPU Memory
     ↓
Insufficient
     ↓
CPU RAM / NVMe Offloading
```

But:

```text
Memory capacity ↑

Training throughput ↓
```

Therefore, my preference would be:

```text
1. Quantization
2. QLoRA
3. Reduce sequence length
4. Gradient checkpointing
5. Distributed sharding
6. CPU offload
```

---

# 23. What I would choose in different situations

| Hardware      | Recommended approach                                            |
| ------------- | --------------------------------------------------------------- |
| 1 × 24GB      | Usually use a smaller model; 70B training is highly constrained |
| 1 × 48GB      | 4-bit QLoRA may be possible for constrained workloads           |
| 2 × 24GB      | QLoRA + sharding                                                |
| 4 × 24GB      | QLoRA + FSDP/ZeRO-3                                             |
| 8 × 24GB      | QLoRA/LoRA + distributed training                               |
| 4–8 × 80GB    | LoRA, QLoRA, or carefully planned full FT                       |
| Large cluster | FSDP/ZeRO full fine-tuning possible                             |

These are planning guidelines, not guarantees; sequence length and implementation details can change requirements significantly.

---

# 24. Production training architecture

```text
                         Dataset
                            │
                            ▼
                    Validation + Cleaning
                            │
                            ▼
                    Tokenization / Packing
                            │
                            ▼
                  Distributed Training Job
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
      GPU 0               GPU 1               GPU N
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                        QLoRA
                            │
                            ▼
                      Checkpoints
                            │
                            ▼
                       Evaluation
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
                Reject              Approve
                                      │
                                      ▼
                                  Model Registry
                                      │
                                      ▼
                                  Deployment
```

I would additionally track:

```text
Dataset version
Base model version
LoRA configuration
Quantization configuration
GPU type/count
Learning rate
Sequence length
Batch size
Training loss
Validation metrics
Tokens/sec
Cost
```

---

# 25. Most important practical point

If you tell me:

```text
"I have only one 24GB GPU and want to fine-tune a 70B model"
```

my honest engineering answer is:

> **I would first question whether 70B is necessary. Training a 70B model with CPU offloading may technically work in some configurations, but the throughput can be so poor that a smaller 7B–14B model, better data, RAG, or distillation may produce a better cost/performance trade-off.**

This is an important interview answer because it demonstrates engineering judgment.

---

# Interview-ready answer

> **For a 70B model with limited GPU resources, I would avoid full fine-tuning because the weights, gradients, optimizer states, and activations exceed the capacity of typical GPUs. I would use QLoRA by loading the base model in 4-bit NF4, freezing the quantized base weights, and training only low-rank LoRA adapters in BF16. I would enable gradient checkpointing, disable the training KV cache, start with a micro-batch size of one, and use gradient accumulation to achieve the desired effective batch size. I would minimize sequence length, use dynamic padding and sequence packing to avoid wasted computation, and use an 8-bit or paged optimizer for the adapters. If multiple GPUs are available, I would use FSDP or DeepSpeed ZeRO-3 to shard memory. CPU offloading would be my last option because it significantly reduces throughput. Finally, I would benchmark tokens per second, memory usage, validation quality, and cost before deciding whether a 70B model is actually justified.**

## Best answer to remember

```text
70B + Limited GPU

        ↓

4-bit Quantization

        ↓

QLoRA

        ↓

Freeze Base Model

        ↓

Train Adapters Only

        ↓

BF16

        ↓

Gradient Checkpointing

        ↓

Small Micro-Batch

        ↓

Gradient Accumulation

        ↓

FSDP / ZeRO-3 if multi-GPU

        ↓

CPU Offloading only as last resort
```

The key distinction for an interview is: **QLoRA reduces trainable-memory requirements, while FSDP/ZeRO reduces per-GPU memory by sharding distributed training state. For a 70B model, you often need one or both depending on available hardware.**
