# Fine-tune an LLM using LoRA — complete practical code

Below is a clean example using:

* Hugging Face `transformers`
* `datasets`
* `peft`
* `trl`
* LoRA
* `SFTTrainer`

We will fine-tune an instruction model on a small customer-support dataset.

---

## 1. Architecture

```text
Training Dataset
       │
       ▼
Tokenization / Chat Template
       │
       ▼
Base LLM
(Frozen weights)
       │
       ▼
LoRA Adapters
(Trainable)
       │
       ▼
Loss
       │
       ▼
Backpropagation
       │
       ▼
Update only LoRA weights
```

Mathematically:

$$
W' = W + \Delta W
$$

where:

$$
\Delta W = \frac{\alpha}{r}BA
$$

During LoRA training:

```text
Base model W      → Frozen ❄️
LoRA matrix A     → Trainable ✓
LoRA matrix B     → Trainable ✓
```

---

# 2. Install dependencies

```bash
pip install torch transformers datasets peft trl accelerate
```

For 4-bit QLoRA support:

```bash
pip install bitsandbytes
```

---

# 3. Create a dataset

Create `train.jsonl`:

```json
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"My payment was deducted twice."},{"role":"assistant","content":"I’m sorry about that. Please share the transaction ID so we can verify the duplicate charge and initiate the appropriate resolution."}]}
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"I forgot my password."},{"role":"assistant","content":"You can reset your password using the Forgot Password option on the login screen."}]}
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"Where is my order?"},{"role":"assistant","content":"Please share your order ID, and I can help check the order status."}]}
```

In a real project, you should have thousands or millions of cleaned examples, not just three.

---

# 4. Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="train.jsonl",
    split="train"
)

print(dataset[0])
```

Example:

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant."
        },
        {
            "role": "user",
            "content": "My payment was deducted twice."
        },
        {
            "role": "assistant",
            "content": "I’m sorry about that..."
        }
    ]
}
```

---

# 5. Split training and validation datasets

Never evaluate only on your training data.

```python
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]
```

You now have:

```text
Dataset
   │
   ├── 90% → Training
   │
   └── 10% → Validation
```

---

# 6. Load the tokenizer

Choose a model you have access to. For example:

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
```

Then:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

---

# 7. Format the conversational dataset

We want:

```text
System
   ↓
User
   ↓
Assistant
```

Use the model's native chat template.

```python
def formatting_func(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }


train_dataset = train_dataset.map(
    formatting_func
)

eval_dataset = eval_dataset.map(
    formatting_func
)
```

Example result:

```python
print(train_dataset[0]["text"])
```

The exact output depends on the model's chat template.

---

# 8. Load the base model

For normal LoRA:

```python
import torch

from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

Enable gradient checkpointing to reduce GPU memory:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

---

# 9. Configure LoRA

This is the most important part.

```python
from peft import (
    LoraConfig,
    get_peft_model
)


lora_config = LoraConfig(

    # Rank
    r=16,

    # Scaling
    lora_alpha=32,

    # Regularization
    lora_dropout=0.05,

    # Target attention projections
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    # Don't train original biases
    bias="none",

    # Causal language model
    task_type="CAUSAL_LM"
)
```

Wrap the base model:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check parameters:

```python
model.print_trainable_parameters()
```

You might see:

```text
trainable params: 8,388,608
all params: 1,500,000,000
trainable%: 0.55%
```

This is the main advantage of LoRA.

Instead of:

```text
1.5 billion trainable parameters
```

you might train:

```text
8 million parameters
```

---

# 10. Configure training

```python
from trl import SFTConfig


training_args = SFTConfig(

    output_dir="./lora-output",

    # Training duration
    num_train_epochs=3,

    # Batch sizes
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    # Effective batch size increases
    gradient_accumulation_steps=8,

    # Learning rate
    learning_rate=2e-4,

    # LR scheduling
    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    # Regularization
    weight_decay=0.01,

    # Precision
    bf16=True,

    # Memory optimization
    gradient_checkpointing=True,

    # Logging
    logging_steps=10,

    # Evaluation
    eval_strategy="steps",
    eval_steps=100,

    # Checkpoints
    save_strategy="steps",
    save_steps=100,

    # Load best checkpoint
    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # Maximum sequence length
    max_length=2048,

    # Logging integration
    report_to="tensorboard"
)
```

### Why is LoRA learning rate often higher?

For full fine-tuning, a common starting range may be around:

```text
1e-6 to 5e-5
```

For LoRA, you are training a small number of newly initialized adapter parameters, so a higher range is often used, for example:

```text
5e-5 to 3e-4
```

But these are starting points, not universal rules.

---

# 11. Create `SFTTrainer`

```python
from trl import SFTTrainer


trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer
)
```

---

# 12. Train

```python
trainer.train()
```

Internally:

```text
Input
  ↓
Base Model
  ↓
LoRA layers
  ↓
Predictions
  ↓
Loss
  ↓
Backpropagation
  ↓
Update LoRA A and B
```

The base model remains frozen.

---

# 13. Monitor training

You might see:

```text
Step 10
loss = 2.41

Step 20
loss = 1.87

Step 30
loss = 1.43
```

Monitor validation:

```text
Step 100

train_loss = 1.20
eval_loss  = 1.31
```

Healthy pattern:

```text
Training loss   ↓
Validation loss ↓
```

Overfitting pattern:

```text
Training loss   ↓
Validation loss ↑
```

---

# 14. Save the LoRA adapter

After training:

```python
ADAPTER_PATH = "./customer-support-lora"

trainer.model.save_pretrained(
    ADAPTER_PATH
)

tokenizer.save_pretrained(
    ADAPTER_PATH
)
```

This generally saves a small adapter artifact rather than another full copy of the base model.

```text
customer-support-lora/
│
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Exact files can vary.

---

# 15. Complete script

Here is the complete example in one file: `train_lora.py`.

```python
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATASET_PATH = "train.jsonl"

OUTPUT_DIR = "./lora-output"

ADAPTER_PATH = "./customer-support-lora"


# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]

eval_dataset = dataset["test"]


# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# FORMAT DATASET
# =========================================================

def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }


train_dataset = train_dataset.map(
    format_example
)

eval_dataset = eval_dataset.map(
    format_example
)


# =========================================================
# LOAD BASE MODEL
# =========================================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

model.config.use_cache = False

model.gradient_checkpointing_enable()


# =========================================================
# CONFIGURE LORA
# =========================================================

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)


# =========================================================
# ADD LORA ADAPTERS
# =========================================================

model = get_peft_model(
    model,
    lora_config
)


# =========================================================
# CHECK TRAINABLE PARAMETERS
# =========================================================

model.print_trainable_parameters()


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    weight_decay=0.01,

    bf16=True,

    gradient_checkpointing=True,

    logging_strategy="steps",

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    max_length=2048,

    report_to="tensorboard"
)


# =========================================================
# CREATE TRAINER
# =========================================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer
)


# =========================================================
# TRAIN
# =========================================================

trainer.train()


# =========================================================
# SAVE ADAPTER
# =========================================================

trainer.model.save_pretrained(
    ADAPTER_PATH
)


# =========================================================
# SAVE TOKENIZER
# =========================================================

tokenizer.save_pretrained(
    ADAPTER_PATH
)


print("LoRA fine-tuning completed successfully!")
```

---

# 16. How to run

```bash
python train_lora.py
```

Or with multiple GPUs:

```bash
accelerate launch train_lora.py
```

---

# 17. Important improvement: response-only training

For conversational fine-tuning, you often do **not** want the model loss to train on the user prompt.

Example:

```text
User:
My payment failed.

Assistant:
I can help you investigate the payment failure.
```

Ideally, you may want the loss to focus primarily on:

```text
Assistant response
```

rather than:

```text
System + User + Assistant
```

Conceptually:

```text
<System>       labels = -100
<User>         labels = -100
<Assistant>    labels = actual tokens
```

This is called **completion-only loss** or **assistant-only loss**, depending on the trainer/template setup.

For chat datasets, configure your training pipeline according to the exact model chat template and TRL version. The details differ across TRL releases, so verify that the assistant tokens are correctly identified before launching a large training job.

---

# 18. Full fine-tuning vs LoRA in this code

### Full fine-tuning

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

for param in model.parameters():

    param.requires_grad = True
```

Result:

```text
All model weights trainable
```

### LoRA

```python
model = get_peft_model(
    model,
    lora_config
)
```

Result:

```text
Base model weights → Frozen
LoRA weights       → Trainable
```

You can verify:

```python
for name, param in model.named_parameters():

    if param.requires_grad:

        print(name)
```

You should see LoRA-related parameters such as:

```text
...q_proj.lora_A...
...q_proj.lora_B...
...v_proj.lora_A...
...v_proj.lora_B...
```

---

# 19. Typical LoRA hyperparameters

A reasonable starting point:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

Experiment with:

| Parameter      | Typical starting values     |
| -------------- | --------------------------- |
| `r`            | 8, 16, 32                   |
| `lora_alpha`   | 16, 32, 64                  |
| `lora_dropout` | 0.0–0.1                     |
| Learning rate  | 5e-5–3e-4                   |
| Epochs         | 1–3 to start                |
| Target modules | Attention projections first |

---

# 20. QLoRA version

If GPU memory is limited, load the base model in 4-bit.

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

Load:

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Then prepare the model for k-bit training:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

Then add LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)
```

The rest of the training code is similar.

Architecture:

```text
Base Model
    ↓
4-bit Quantization
    ↓
Frozen Base Weights
    +
BF16/FP16 LoRA Adapters
    ↓
QLoRA Training
```

---

# Interview-ready explanation

> **To fine-tune an LLM with LoRA, I first load a pretrained base model and tokenizer. I freeze the base model and configure LoRA adapters on selected transformer projection layers such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The LoRA configuration defines the rank, scaling factor, dropout, and target modules.**
>
> **I then use a trainer such as TRL's `SFTTrainer` to train the instruction or conversational dataset. During backpropagation, only the LoRA adapter parameters are updated, which significantly reduces the number of trainable parameters and GPU memory compared with full fine-tuning.**
>
> **I monitor training loss and validation loss, save the best checkpoint, and finally save the adapter separately using `save_pretrained()`. During inference, I load the same base model and attach the LoRA adapter using PEFT.**

The essential code flow is:

```text
Load Dataset
      ↓
Load Base Model
      ↓
Create LoraConfig
      ↓
get_peft_model()
      ↓
SFTTrainer
      ↓
Train
      ↓
Save Adapter
```
