# How would you fine-tune a 70B LLM?

Fine-tuning a **70B parameter model** is very different from fine-tuning a 7B model.

The first question is:

> **Do we really want full fine-tuning?**

Usually:

```text
70B model
    ↓
Full fine-tuning → extremely expensive
    ↓
LoRA / QLoRA → usually preferred
```

For most real-world enterprise use cases, I would use:

```text
70B Base Model
+
4-bit Quantization
+
QLoRA
+
Multiple GPUs
+
FSDP or DeepSpeed
+
BF16 compute
+
Gradient Checkpointing
```

---

# 1. First understand the memory problem

A 70B model has approximately:

```text
70,000,000,000 parameters
```

## Model weights alone

### FP32

```text
70B × 4 bytes

≈ 280 GB
```

### FP16 / BF16

```text
70B × 2 bytes

≈ 140 GB
```

### 4-bit quantization

```text
70B × 0.5 bytes

≈ 35 GB
```

But training needs much more than just model weights.

You also need:

```text
Model parameters
+
Gradients
+
Optimizer states
+
Activations
+
Temporary tensors
```

Therefore, full fine-tuning can require hundreds of GB or more depending on precision and optimizer.

---

# 2. Best practical approach: QLoRA

For most companies, I would **not** fully fine-tune the 70B model.

I would use:

```text
                    70B Base Model
                          │
                          ▼
                 Load in 4-bit format
                          │
                          ▼
                    Freeze weights
                          │
                          ▼
                Add LoRA adapters
                          │
                          ▼
              Train only adapters
```

Instead of training:

```text
70,000,000,000 parameters
```

you might train only a relatively small number of LoRA parameters.

This dramatically reduces memory requirements.

---

# 3. Recommended architecture

A production training architecture could look like:

```text
                 Training Dataset
                       │
                       ▼
              Data Validation
                       │
                       ▼
                 Tokenization
                       │
                       ▼
        ┌─────────────────────────┐
        │ 70B Base Model          │
        │                         │
        │ 4-bit Quantization      │
        │                         │
        │ Frozen Base Weights     │
        └───────────┬─────────────┘
                    │
                    ▼
             LoRA Adapters
                    │
                    ▼
              Train Adapters
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
   DeepSpeed                  FSDP
   ZeRO                       Sharding
        │                        │
        └───────────┬────────────┘
                    ▼
                 Checkpoint
                    │
                    ▼
               Evaluation
                    │
                    ▼
              Model Registry
```

---

# 4. Hardware options

A realistic setup might be:

### Option 1: Large GPUs

```text
8 × A100 80GB
```

or:

```text
8 × H100 80GB
```

### Option 2: QLoRA

Potentially fewer GPUs can work depending on:

```text
Sequence length
Batch size
Dataset
Model architecture
Quantization implementation
Activation memory
```

For a serious production job, I would generally prefer multiple high-memory GPUs rather than designing around the absolute minimum hardware.

---

# 5. Install libraries

```bash
pip install torch transformers datasets accelerate
pip install peft bitsandbytes trl
```

For distributed training:

```bash
pip install deepspeed
```

---

# 6. Prepare the instruction dataset

Example:

```json
{
    "instruction": "Explain how to reset my password.",
    "input": "",
    "output": "To reset your password, open the login page and click Forgot Password..."
}
```

A JSONL dataset:

```json
{"instruction":"Explain Python decorators","input":"","output":"A decorator modifies the behavior of a function..."}
{"instruction":"Explain RAG","input":"","output":"RAG combines retrieval with generation..."}
```

---

# 7. Load the dataset

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "test": "test.jsonl"
    }
)
```

Check:

```python
print(dataset)
```

---

# 8. Format the dataset

For an instruction-following model:

```python
def format_example(example):

    prompt = f"""
### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}
"""

    return {
        "text": prompt
    }
```

Apply:

```python
dataset = dataset.map(
    format_example
)
```

For a chat model, use its tokenizer's chat template whenever available instead of inventing a format.

---

# 9. Load the tokenizer

```python
from transformers import AutoTokenizer


MODEL_NAME = "your-70b-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
```

Set padding:

```python
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token
```

---

# 10. Load the 70B model in 4-bit

This is the core QLoRA technique.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
```

Configure 4-bit quantization:

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Load the model:

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

Conceptually:

```text
Original model

70B parameters
      │
      ▼
4-bit quantization
      │
      ▼
Base model mostly frozen
      │
      ▼
Train LoRA adapters
```

---

# 11. Prepare the model for QLoRA

```python
from peft import (
    prepare_model_for_kbit_training
)


model = prepare_model_for_kbit_training(
    model
)
```

This prepares the quantized model for adapter-based training.

Enable gradient checkpointing:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

---

# 12. Configure LoRA

Now add trainable LoRA adapters.

```python
from peft import (
    LoraConfig
)
```

Example:

```python
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
```

For a more aggressive adaptation, you might also target MLP projections, but the exact module names must match the model architecture.

For example:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

---

# 13. Why target these layers?

A transformer looks conceptually like:

```text
Transformer Block

Input
  │
  ▼
Attention
  │
  ├── Q Projection
  ├── K Projection
  ├── V Projection
  └── O Projection
  │
  ▼
MLP
  │
  ├── Gate Projection
  ├── Up Projection
  └── Down Projection
```

LoRA adds small trainable matrices.

Instead of:

```text
W
```

we approximate an update:

```text
W + ΔW
```

where:

```text
ΔW = B × A
```

`W` remains frozen.

Only:

```text
A
B
```

are trained.

---

# 14. Create the PEFT model

```python
from peft import (
    get_peft_model
)


model = get_peft_model(
    model,
    lora_config
)
```

Check trainable parameters:

```python
model.print_trainable_parameters()
```

You should see something conceptually like:

```text
Trainable params:
small percentage

Total params:
70B
```

This is the main advantage of QLoRA.

---

# 15. Configure training

Using `SFTTrainer` is convenient for supervised instruction fine-tuning.

```python
from transformers import TrainingArguments
```

Example:

```python
training_args = TrainingArguments(

    output_dir="./70b-support-model",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,

    bf16=True,

    logging_steps=10,

    save_steps=500,

    save_total_limit=3,

    eval_strategy="steps",

    eval_steps=500,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    report_to="none"
)
```

---

# 16. Why batch size = 1?

A 70B model consumes a large amount of memory.

Instead of:

```text
Batch size = 16
```

we might use:

```text
Micro batch size = 1

Gradient accumulation = 16
```

Conceptually:

```text
Batch 1
   ↓
Calculate gradients
   ↓
Don't update

Batch 2
   ↓
Calculate gradients
   ↓
Don't update

...

Batch 16
   ↓
Calculate gradients
   ↓
Optimizer update
```

Effective batch size per process:

```text
1 × 16 = 16
```

With multiple GPUs, the global effective batch size also depends on the number of processes.

For example:

```text
per_device_batch
× gradient_accumulation
× number_of_GPUs
```

---

# 17. Full QLoRA training example

Here is a simplified end-to-end example.

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
```

## Configuration

```python
MODEL_NAME = "your-70b-model"

OUTPUT_DIR = "./output-70b"
```

---

## Dataset

```python
dataset = load_dataset(

    "json",

    data_files={
        "train": "train.jsonl",
        "test": "test.jsonl"
    }
)
```

Format:

```python
def format_example(example):

    return {
        "text": f"""
### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}
"""
    }
```

```python
dataset = dataset.map(
    format_example
)
```

---

## Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
```

---

## Quantization

```python
bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

---

## Model

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

Prepare for training:

```python
model = prepare_model_for_kbit_training(
    model
)

model.gradient_checkpointing_enable()

model.config.use_cache = False
```

---

## LoRA configuration

```python
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

---

## Training arguments

```python
training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,

    bf16=True,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=500,

    save_steps=500,

    save_total_limit=3,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    warmup_ratio=0.03,

    lr_scheduler_type="cosine",

    report_to="none"
)
```

---

## Create the trainer

Recent TRL versions may use a slightly different `SFTTrainer` dataset/config API, so the exact constructor can vary by installed version. The underlying approach remains the same.

A common pattern is:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"],

    peft_config=lora_config
)
```

Train:

```python
trainer.train()
```

---

# 18. Save the LoRA adapter

After training:

```python
trainer.save_model(
    "./70b-lora-adapter"
)
```

You will typically get:

```text
70b-lora-adapter/

├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

You do **not** normally save another full 70B copy.

That is one major benefit of LoRA.

---

# 19. Loading the adapter for inference

Load the base model:

```python
base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

Load the adapter:

```python
from peft import PeftModel


model = PeftModel.from_pretrained(

    base_model,

    "./70b-lora-adapter"
)
```

Set evaluation mode:

```python
model.eval()
```

---

# 20. Run inference

```python
prompt = """
### Instruction:
Explain what RAG is.

### Response:
"""
```

Tokenize:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)
```

Move tensors to the appropriate device:

```python
inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.no_grad():

    output = model.generate(

        **inputs,

        max_new_tokens=300,

        temperature=0.7,

        do_sample=True
    )
```

Decode:

```python
response = tokenizer.decode(

    output[0],

    skip_special_tokens=True
)

print(response)
```

---

# 21. Using DeepSpeed for a 70B model

For multi-GPU full training or larger distributed workloads, DeepSpeed can help.

Example `ds_zero3.json`:

```json
{
    "bf16": {
        "enabled": true
    },

    "gradient_accumulation_steps": "auto",

    "train_micro_batch_size_per_gpu": "auto",

    "zero_optimization": {

        "stage": 3,

        "overlap_comm": true,

        "contiguous_gradients": true
    }
}
```

Then:

```python
training_args = TrainingArguments(

    output_dir="./70b-output",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    bf16=True,

    gradient_checkpointing=True,

    deepspeed="./ds_zero3.json"
)
```

Conceptually:

```text
                    70B Model

                         │

                         ▼

                    ZeRO-3

                         │

        ┌────────────────┼────────────────┐

        ▼                ▼                ▼

      GPU 0            GPU 1            GPU 2

   Param shard A     Param shard B     Param shard C

   Grad shard A      Grad shard B      Grad shard C

   Opt shard A       Opt shard B       Opt shard C
```

---

# 22. But can you combine QLoRA and DeepSpeed ZeRO-3?

This requires care.

This is an important practical point.

```text
QLoRA
=
Quantized frozen base model
+
Trainable adapters
```

```text
ZeRO-3
=
Distributed parameter/state sharding
```

They can be used in distributed training setups, but the exact configuration must be compatible with:

```text
Transformers version
PEFT version
TRL version
DeepSpeed version
bitsandbytes version
Quantized loading strategy
```

For an interview, I would say:

> For a 70B model, I would first prefer QLoRA because it drastically reduces trainable memory. If one GPU is insufficient, I would use distributed training and choose the parallelism strategy based on the model-loading and training stack. For full fine-tuning, ZeRO-3 or FSDP becomes much more important.

---

# 23. Full fine-tuning a 70B model

If you truly need to update all 70B parameters:

```text
70B model

Full parameters
+
Full gradients
+
Optimizer states
+
Activations
```

You would likely use:

```text
Many high-memory GPUs

+
FSDP FULL_SHARD
or
DeepSpeed ZeRO-3

+
BF16

+
Activation checkpointing

+
Flash Attention

+
Gradient accumulation
```

Conceptually:

```text
                  70B LLM

                     │

              Full Fine-Tuning

                     │

        ┌────────────┼─────────────┐

        ▼            ▼             ▼

      FSDP         BF16        Checkpointing

        │            │             │

        └────────────┼─────────────┘

                     ▼

                Multi-GPU
```

---

# 24. QLoRA vs full fine-tuning for 70B

| Feature              | Full Fine-Tuning                                      | QLoRA                         |
| -------------------- | ----------------------------------------------------- | ----------------------------- |
| Base model weights   | Trainable                                             | Frozen                        |
| Trainable parameters | ~70B                                                  | Small adapters                |
| Memory requirement   | Very high                                             | Much lower                    |
| Cost                 | Very high                                             | Lower                         |
| Training speed       | Depends on cluster                                    | Often easier to make feasible |
| Best for             | Large behavior/domain changes with sufficient compute | Most domain/task adaptation   |
| Storage              | Full model checkpoint                                 | Small adapter checkpoints     |

---

# 25. What would I do in a real company?

My approach would be:

## Step 1: Don't immediately fine-tune

First check:

```text
Can prompt engineering solve it?

Can RAG solve it?

Does the problem require model behavior changes?
```

For example:

```text
Company documents
        ↓
Use RAG
```

```text
Specific response format
        ↓
Prompting / structured output
```

```text
Consistent company behavior
        ↓
Fine-tuning
```

---

## Step 2: Start with a smaller model

I would benchmark:

```text
8B model
14B model
70B model
```

against:

```text
Accuracy
Latency
Cost
Throughput
```

A 70B model is not automatically the best production choice.

---

## Step 3: Start with QLoRA

For a 70B model:

```text
4-bit base model
+
NF4
+
BF16 compute
+
LoRA
+
Gradient checkpointing
```

---

## Step 4: Scale only if required

```text
Single GPU
    ↓
Not enough?
    ↓
Multi-GPU
    ↓
Accelerate / torchrun
    ↓
FSDP or DeepSpeed
```

---

# 26. Production training workflow

A real production pipeline would look like:

```text
                    Raw Dataset
                         │
                         ▼
                 Data Validation
                         │
                         ▼
                Remove Bad Samples
                         │
                         ▼
                 Dataset Versioning
                         │
                         ▼
                   Train / Eval Split
                         │
                         ▼
                 70B QLoRA Training
                         │
                         ▼
                  Experiment Tracking
                         │
                    MLflow/W&B
                         │
                         ▼
                   Evaluation
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Quality      Safety     Latency
             │           │           │
             └───────────┼───────────┘
                         ▼
                  Human Evaluation
                         │
                         ▼
                   Register Adapter
                         │
                         ▼
                      Deploy
```

---

# Interview-ready answer

> **For fine-tuning a 70B model, I would first avoid full fine-tuning unless there is a strong business requirement because the GPU memory and infrastructure cost are very high. My default approach would be QLoRA: load the base model in 4-bit NF4, freeze the quantized base weights, add LoRA adapters to the attention and optionally MLP projection layers, and train only the adapters using BF16 computation, gradient checkpointing, and gradient accumulation.**
>
> **If one GPU is insufficient, I would scale to multiple GPUs using Accelerate, FSDP, or DeepSpeed depending on the training stack. For full fine-tuning, I would typically use multi-GPU FSDP FULL_SHARD or DeepSpeed ZeRO-3 with BF16 and activation checkpointing.**
>
> **After training, I would save the LoRA adapter separately, evaluate it against the base model using task-specific metrics and human evaluation, version the dataset and experiment, register the resulting adapter, and deploy it with the appropriate base model.**

## The key architecture to remember

```text
70B Model
    │
    ▼
4-bit Quantization
    │
    ▼
Freeze Base Model
    │
    ▼
Add LoRA Adapters
    │
    ▼
QLoRA Training
    │
    ├── BF16
    ├── Gradient Checkpointing
    ├── Gradient Accumulation
    └── Multi-GPU if needed
    │
    ▼
Save Adapter
    │
    ▼
Evaluate
    │
    ▼
Deploy
```

**The most important interview point:**

> **For a 70B model, my default production choice is usually QLoRA rather than full fine-tuning, because I can adapt the model while training only a small number of parameters and keeping the massive base model frozen.**
