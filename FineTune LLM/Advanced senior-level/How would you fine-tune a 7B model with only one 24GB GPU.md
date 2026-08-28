# How would you fine-tune a 7B model with only one 24GB GPU?

Yes. The practical answer is:

> **I would not do full fine-tuning. On a single 24GB GPU, I would usually use QLoRA: load the 7B base model in 4-bit, keep the base weights frozen, train LoRA adapters in BF16, use gradient checkpointing, a small micro-batch, gradient accumulation, and dynamic padding/packing.**

A typical setup is:

```text
24 GB GPU
   │
   ├── 7B model loaded in 4-bit
   ├── LoRA adapters trainable
   ├── BF16 computation
   ├── Gradient checkpointing
   ├── Micro-batch size: 1–4
   └── Gradient accumulation
```

---

# 1. Why full fine-tuning is difficult on 24GB

A 7B model has approximately:

```text
7 billion parameters
```

Just the weights in FP16:

```text
7B × 2 bytes
≈ 14 GB
```

But training requires much more:

```text
Model weights
+ Gradients
+ Optimizer states
+ Activations
+ Temporary tensors
```

A simplified view:

```text
FP16/BF16 weights     ≈ 14 GB
Gradients             ≈ 14 GB
Optimizer states      ≈ 56 GB+ depending on optimizer/precision
Activations           variable
--------------------------------
Total                 far above 24 GB
```

So:

```text
❌ Full fine-tuning on one 24GB GPU
```

is generally impractical for a standard Adam-based setup.

Instead:

```text
7B Base Model
      │
      ▼
4-bit Quantization
      │
      ▼
Frozen Base Weights
      +
Train LoRA Adapters
      │
      ▼
QLoRA
```

---

# 2. Architecture of QLoRA

```text
                Base LLM

        ┌────────────────────┐
        │      7B Model      │
        │                    │
        │  W = frozen        │
        │  loaded in 4-bit   │
        └─────────┬──────────┘
                  │
                  ▼
             LoRA layers

            A          B
            ↓          ↓

        Trainable small matrices

                  │
                  ▼

             Loss
                  │
                  ▼

       Update LoRA adapters only
```

The base model remains frozen.

Instead of training:

```text
7,000,000,000 parameters
```

you may train only:

```text
tens of millions of adapter parameters
```

depending on rank and target modules.

---

# 3. Recommended stack

For example:

```text
Python
PyTorch
Transformers
Datasets
PEFT
TRL
bitsandbytes
Accelerate
```

Install:

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

For production work, **pin versions after validating the stack** because CUDA, PyTorch, Transformers, PEFT, TRL, and bitsandbytes compatibility matters.

---

# 4. Project structure

A clean structure:

```text
llm-finetuning/
│
├── data/
│   └── train.jsonl
│
├── train.py
├── inference.py
├── requirements.txt
│
├── checkpoints/
└── adapters/
```

---

# 5. Prepare the dataset

Example:

```json
{"instruction":"What is RAG?","input":"","output":"RAG stands for Retrieval-Augmented Generation. It retrieves relevant external information before generating an answer."}
{"instruction":"Explain LoRA.","input":"","output":"LoRA is a parameter-efficient fine-tuning technique that trains low-rank adapter matrices while keeping the base model frozen."}
```

This is stored as:

```text
data/train.jsonl
```

---

# 6. Format the dataset

We need to convert:

```text
Instruction
+
Input
+
Output
```

into a training example.

```python
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

    return {
        "text": text
    }
```

Load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)

dataset = dataset["train"]

dataset = dataset.map(
    format_example
)
```

---

# 7. Load the tokenizer

```python
from transformers import AutoTokenizer

MODEL_NAME = "your-7b-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

For causal language models:

```python
tokenizer.padding_side = "right"
```

Right padding is commonly used during training.

---

# 8. Configure 4-bit quantization

This is the most important part.

```python
import torch

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(

    # Load base model in 4-bit
    load_in_4bit=True,

    # NF4 is commonly used for QLoRA
    bnb_4bit_quant_type="nf4",

    # Quantize quantization constants
    bnb_4bit_use_double_quant=True,

    # Compute in BF16
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Why?

```text
Without quantization:

7B × FP16
≈ 14 GB model weights
```

With approximate 4-bit storage:

```text
7B × 0.5 bytes
≈ 3.5 GB theoretical raw weight storage
```

Real memory usage is higher because of quantization metadata and runtime overhead, but the reduction is still substantial.

---

# 9. Load the model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

For a single-GPU training job, you generally want the training stack to manage device placement consistently. If you run into device-placement issues, avoid blindly mixing `device_map="auto"` with distributed training frameworks.

---

# 10. Prepare the model for QLoRA training

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

This prepares the quantized model for parameter-efficient training.

Now enable gradient checkpointing:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

Why disable cache?

During training:

```text
KV cache
+
Gradient checkpointing
```

is generally unnecessary and can cause compatibility/memory issues.

---

# 11. Configure LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(

    # LoRA rank
    r=16,

    # Scaling
    lora_alpha=32,

    # Attention layers
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    # Regularization
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

The exact `target_modules` depend on the model architecture.

For example, many Llama-style architectures use names such as:

```text
q_proj
k_proj
v_proj
o_proj
```

You can inspect the model:

```python
for name, module in model.named_modules():

    if "proj" in name:
        print(name)
```

---

# 12. Check how many parameters are trainable

Apply LoRA:

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)
```

Then:

```python
model.print_trainable_parameters()
```

You may see something conceptually like:

```text
trainable params: 30,000,000
all params: 7,000,000,000

trainable: 0.4%
```

The exact number depends on:

```text
LoRA rank
Target modules
Model architecture
```

This is why LoRA is practical on smaller GPUs.

---

# 13. Configure training arguments for 24GB

A good starting point:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(

    output_dir="./checkpoints",

    # Small micro-batch
    per_device_train_batch_size=1,

    # Simulate larger batch
    gradient_accumulation_steps=16,

    # Training
    num_train_epochs=3,

    learning_rate=2e-4,

    # Mixed precision
    bf16=True,

    # Logging
    logging_steps=10,

    # Checkpointing
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    # Performance
    optim="paged_adamw_8bit",

    # Reporting
    report_to="none",

    # Reproducibility
    seed=42
)
```

Effective batch size:

```text
micro batch
×
gradient accumulation
×
number of GPUs
```

Here:

```text
1 × 16 × 1 = 16
```

---

# 14. Use SFTTrainer

With modern TRL, APIs can vary by version, but the basic idea is:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,

    train_dataset=dataset,

    args=training_args,

    peft_config=lora_config
)
```

Depending on the installed TRL version, you may need to specify the text field or provide preprocessing explicitly.

For example, in some versions:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=2048
)
```

Because TRL evolves quickly, check the installed version's API rather than copying arguments blindly.

---

# 15. Complete practical training script

Here is a consolidated QLoRA example:

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
# 1. Configuration
# ============================================================

MODEL_NAME = "your-7b-model"

OUTPUT_DIR = "./output"

MAX_SEQ_LENGTH = 2048


# ============================================================
# 2. Load dataset
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

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)


# ============================================================
# 3. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4. 4-bit QLoRA configuration
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)


# ============================================================
# 5. Load model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# ============================================================
# 6. Prepare for k-bit training
# ============================================================

model = prepare_model_for_kbit_training(
    model
)

model.gradient_checkpointing_enable()

model.config.use_cache = False


# ============================================================
# 7. LoRA configuration
# ============================================================

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


# ============================================================
# 8. Attach LoRA adapters
# ============================================================

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ============================================================
# 9. Training configuration
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    learning_rate=2e-4,

    bf16=True,

    logging_steps=10,

    save_strategy="steps",

    save_steps=200,

    save_total_limit=2,

    optim="paged_adamw_8bit",

    report_to="none",

    seed=42
)


# ============================================================
# 10. Trainer
# ============================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    args=training_args,

    dataset_text_field="text",

    max_seq_length=MAX_SEQ_LENGTH,

    tokenizer=tokenizer
)


# ============================================================
# 11. Train
# ============================================================

trainer.train()


# ============================================================
# 12. Save adapters
# ============================================================

trainer.save_model(
    "./lora_adapter"
)

tokenizer.save_pretrained(
    "./lora_adapter"
)
```

---

# 16. Important: dataset formatting should match inference

Suppose training data uses:

```text
### Instruction:
Explain RAG.

### Response:
RAG is...
```

During inference, use the same style:

```python
prompt = """### Instruction:
Explain RAG.

### Response:
"""
```

Then:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(

        **inputs,

        max_new_tokens=200,

        do_sample=False
    )
```

Decode:

```python
print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)
```

---

# 17. Better: use a validation dataset

Don't train on everything.

Split:

```python
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]

eval_dataset = dataset["test"]
```

Then configure evaluation.

The exact `TrainingArguments` evaluation options vary slightly across library versions, but conceptually:

```python
TrainingArguments(
    ...,
    eval_strategy="steps",
    eval_steps=200
)
```

Then:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args
)
```

Monitor:

```text
Training loss
Validation loss
Task-specific quality
```

---

# 18. How much VRAM should I expect?

This depends heavily on:

```text
Model architecture
Context length
Batch size
Flash Attention
LoRA targets
Optimizer
CUDA/PyTorch versions
```

But conceptually:

```text
24 GB GPU
│
├── Quantized 7B base model
├── LoRA parameters
├── Activations
├── Optimizer states for LoRA
├── CUDA kernels
└── Memory overhead
```

A 7B QLoRA setup with:

```text
4-bit base
BF16 compute
LoRA r=16
batch=1
gradient accumulation
sequence length ~1024–2048
```

is generally a realistic starting point for a 24GB GPU.

If OOM occurs:

```text
First reduce:
1. Sequence length
2. Micro-batch size
3. LoRA target modules/rank
```

Sequence length often has a major effect on activation memory.

---

# 19. A practical configuration ladder

I would start with:

```python
MAX_SEQ_LENGTH = 1024

per_device_train_batch_size = 1

gradient_accumulation_steps = 16

r = 16
```

If stable:

```text
1024 → 1536 → 2048
```

Benchmark:

```text
tokens/sec
GPU memory
training loss
validation quality
```

Then tune.

For example:

```text
Config A
Seq = 1024
Batch = 1
VRAM = 14 GB

Config B
Seq = 2048
Batch = 1
VRAM = 19 GB

Config C
Seq = 4096
Batch = 1
VRAM = 27 GB → OOM
```

Then I would select:

```text
2048
```

if it gives sufficient task quality.

---

# 20. What I would do if QLoRA still OOMs

In this order:

### 1. Reduce sequence length

```python
MAX_SEQ_LENGTH = 1024
```

### 2. Keep micro-batch at 1

```python
per_device_train_batch_size = 1
```

### 3. Increase gradient accumulation

```python
gradient_accumulation_steps = 32
```

### 4. Enable gradient checkpointing

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
```

### 5. Reduce LoRA rank

```python
r=8
```

instead of:

```text
r=32 or r=64
```

### 6. Reduce target modules

For example:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

### 7. Check for memory leaks

Avoid:

```python
losses.append(loss)
```

Use:

```python
losses.append(loss.item())
```

---

# 21. A more production-oriented configuration

For a real training job on a single 24GB GPU:

```python
training_args = TrainingArguments(

    output_dir="./checkpoints",

    # Memory
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,

    # Training
    num_train_epochs=3,
    learning_rate=2e-4,

    # Optimization
    optim="paged_adamw_8bit",

    # Checkpointing
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # Logging
    logging_steps=10,

    # Data
    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    # Reliability
    seed=42,

    report_to="none"
)
```

I would also monitor:

```text
GPU memory
GPU utilization
tokens/sec
training loss
validation loss
checkpoint size
```

---

# 22. Interview-ready architecture

```text
                    JSONL Dataset
                         │
                         ▼
                   Format prompts
                         │
                         ▼
                    Tokenization
                         │
                         ▼
              ┌─────────────────────┐
              │     7B LLM          │
              │                     │
              │  4-bit NF4 weights  │
              │       FROZEN        │
              └──────────┬──────────┘
                         │
                         ▼
                  LoRA Adapters
                  r=8 / 16 / 32
                         │
                         ▼
                    BF16 Compute
                         │
                         ▼
              Gradient Checkpointing
                         │
                         ▼
              Paged 8-bit Optimizer
                         │
                         ▼
                  LoRA Adapter
                     Saved
```

---

# Best interview answer

> **For a 7B model on a single 24GB GPU, I would use QLoRA rather than full fine-tuning. I would load the base model in 4-bit NF4 using bitsandbytes, keep the base weights frozen, and train only LoRA adapters. I would use BF16 for computation, gradient checkpointing to reduce activation memory, and disable the KV cache during training. I would start with a micro-batch size of 1 and use gradient accumulation to achieve the desired effective batch size. I would use dynamic padding or sequence packing to reduce wasted computation and keep the maximum sequence length as low as the task allows. For optimization, I would use an 8-bit or paged optimizer. Finally, I would monitor GPU memory, tokens per second, training loss, and validation quality, then tune sequence length and batch size based on measured results.**

### The configuration I would remember for interviews

```text
7B model
+
4-bit NF4
+
QLoRA
+
BF16 compute
+
Gradient checkpointing
+
Batch size = 1–4
+
Gradient accumulation
+
Sequence length = 1024–2048
+
Paged/8-bit optimizer
=
Fine-tuning on one 24GB GPU
```

This is the standard practical approach for your kind of **Senior AI Engineer interview** question.
