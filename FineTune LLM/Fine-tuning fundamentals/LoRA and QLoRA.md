# LoRA and QLoRA — Explained Properly with Code

These are two of the most important techniques for fine-tuning LLMs efficiently.

---

# 1. What is LoRA?

**LoRA (Low-Rank Adaptation)** is a **Parameter-Efficient Fine-Tuning (PEFT)** technique.

Instead of updating every weight in a large language model, LoRA:

1. Freezes the original model weights.
2. Adds small trainable matrices to selected layers.
3. Trains only those small matrices.

```text
                Full Fine-Tuning

Base Model
    │
    ▼
Update all parameters
    │
    ▼
Expensive


                    LoRA

Base Model
    │
    ├── Freeze original weights ❄️
    │
    └── Add small trainable matrices
                  │
                  ▼
              Train only them
```

---

# 2. Why do we need LoRA?

Suppose an LLM has:

```text
7 billion parameters
```

With full fine-tuning:

```text
7B parameters
    +
Gradients
    +
Optimizer states
    +
Activations
```

This requires substantial GPU memory.

But for many tasks, we may not need to change the entire model.

We only need to learn a task-specific adjustment.

LoRA assumes the required weight update can often be represented approximately using a **low-rank decomposition**.

---

# 3. LoRA mathematically

Suppose a Transformer has a weight matrix:

```text
W
```

For example:

```text
W ∈ R^(d × k)
```

During full fine-tuning:

```text
W → W + ΔW
```

We need to learn the entire:

```text
ΔW
```

If:

```text
W = 4096 × 4096
```

then:

```text
16,777,216 parameters
```

could potentially be updated for that matrix.

---

## LoRA idea

Instead of directly learning:

```text
ΔW
```

LoRA approximates the update as:

```text
ΔW = B × A
```

where:

```text
A ∈ R^(r × k)

B ∈ R^(d × r)

r << d, k
```

For example:

```text
Original matrix:

W = 4096 × 4096

Rank:

r = 8
```

LoRA matrices:

```text
A = 8 × 4096
B = 4096 × 8
```

Total trainable parameters:

```text
8 × 4096 + 4096 × 8

= 32,768 + 32,768

= 65,536
```

Compare:

```text
Full update:
16,777,216 parameters

LoRA:
65,536 parameters
```

So LoRA dramatically reduces trainable parameters.

---

# 4. LoRA forward pass

Original layer:

```text
Y = XW
```

LoRA adds an update:

```text
Y = X(W + ΔW)
```

Since:

```text
ΔW = BA
```

we get:

```text
Y = XW + XBA
```

Often, LoRA applies scaling:

```text
Y = XW + (α / r) × XBA
```

Where:

* `W` = original frozen weight
* `A` and `B` = trainable LoRA matrices
* `r` = rank
* `α` = scaling factor

---

# 5. Visual representation

```text
                   Input X
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
         Base Weight W       LoRA A
           Frozen ❄️            │
              │                 ▼
              │              LoRA B
              │                 │
              └────────┬────────┘
                       ▼
                     Add
                       │
                       ▼
                    Output
```

During training:

```text
W → Frozen ❄️

A → Trainable ✓
B → Trainable ✓
```

---

# 6. Where is LoRA applied in an LLM?

A Transformer contains many linear layers.

For attention:

```text
Input
  │
  ├── Wq → Query
  │
  ├── Wk → Key
  │
  ├── Wv → Value
  │
  └── Wo → Output
```

LoRA is often applied to selected projection layers.

For example:

```text
Wq → Wq + ΔWq

Wv → Wv + ΔWv
```

In many LLaMA-style models:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

You might also adapt MLP layers:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

The exact layer names depend on the model architecture.

---

# 7. LoRA code from scratch

Let's understand the mathematics with a small PyTorch example.

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        alpha=16
    ):
        super().__init__()

        # Original pretrained weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        # Freeze base weight
        self.weight.requires_grad = False

        # LoRA A
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )

        # LoRA B
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.scaling = alpha / rank


    def forward(self, x):

        # Base model output
        base_output = x @ self.weight.T

        # LoRA update
        lora_output = (
            x
            @ self.lora_A.T
            @ self.lora_B.T
        ) * self.scaling

        return base_output + lora_output
```

Use it:

```python
layer = LoRALinear(
    in_features=4,
    out_features=8,
    rank=2,
    alpha=4
)

x = torch.randn(2, 4)

output = layer(x)

print(output.shape)
```

Output:

```text
torch.Size([2, 8])
```

---

# 8. Which parameters are trainable?

```python
for name, parameter in layer.named_parameters():

    print(
        name,
        parameter.requires_grad
    )
```

Expected:

```text
weight False

lora_A True

lora_B True
```

So:

```text
Base Weight
   ↓
Frozen ❄️

LoRA A
   ↓
Trainable ✓

LoRA B
   ↓
Trainable ✓
```

---

# 9. Why initialize B with zeros?

Notice:

```python
self.lora_B = nn.Parameter(
    torch.zeros(out_features, rank)
)
```

Initially:

```text
B = 0
```

Therefore:

```text
BA = 0
```

So initially:

```text
W_new = W + BA

W_new = W
```

This means training starts with the original pre-trained model behavior.

Then:

```text
Training
    ↓
Gradients
    ↓
B and A change
    ↓
LoRA contribution increases
```

---

# 10. LoRA using Hugging Face PEFT

Now the practical implementation.

## Install

```bash
pip install torch transformers datasets peft accelerate
```

---

## Load model

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

---

## Configure LoRA

```python
from peft import (
    LoraConfig,
    get_peft_model
)


lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

Apply LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Conceptually:

```text
trainable params: 500,000
all params: 124,000,000
trainable: 0.4%
```

Exact numbers depend on the model and configuration.

---

# 11. Complete LoRA fine-tuning example

Dataset:

```python
from datasets import Dataset


data = {
    "instruction": [
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue."
    ],

    "input": [
        "I was charged twice.",
        "I cannot log in.",
        "The application crashes."
    ],

    "output": [
        "billing",
        "account_access",
        "technical"
    ]
}

dataset = Dataset.from_dict(data)
```

Format and tokenize:

```python
MAX_LENGTH = 256


def tokenize_example(example):

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    response = example["output"] + tokenizer.eos_token

    full_text = prompt + response

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = full_tokens["input_ids"].copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # Do not calculate loss on the prompt
    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    full_tokens["labels"] = labels

    return full_tokens
```

Apply:

```python
tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)
```

Train:

```python
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)


training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    report_to="none"
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


trainer.train()
```

Save only the adapter:

```python
model.save_pretrained(
    "./lora_adapter"
)

tokenizer.save_pretrained(
    "./lora_adapter"
)
```

---

# 12. What is QLoRA?

**QLoRA = Quantized LoRA.**

QLoRA combines:

```text
Quantization
       +
LoRA
       =
QLoRA
```

The idea:

```text
Large Base Model
      ↓
Quantize model weights
      ↓
Usually 4-bit representation
      ↓
Freeze quantized base model
      +
Train LoRA adapters
```

So:

```text
             QLoRA

      Base Model
           │
           ▼
      Quantization
       16-bit/32-bit
            ↓
          4-bit
           │
       Frozen ❄️
           +
      LoRA Adapters
       Trainable ✓
```

---

# 13. Why QLoRA?

Suppose you want to fine-tune a 7B model.

Normal model loading might require significant GPU memory.

QLoRA reduces memory used by the frozen base weights.

```text
LoRA:

Base Model
higher precision
+
LoRA


QLoRA:

Base Model
4-bit quantized
+
LoRA
```

This allows fine-tuning larger models on more limited hardware.

---

# 14. Important QLoRA concept

A common misconception is:

> "QLoRA trains the entire model in 4-bit."

Not exactly.

Conceptually:

```text
Base Model
   ↓
Stored quantized (commonly 4-bit)
   ↓
Frozen

LoRA adapters
   ↓
Higher-precision trainable parameters
   ↓
Updated during training
```

The exact compute and storage dtypes depend on the implementation and hardware.

---

# 15. QLoRA code

Install:

```bash
pip install torch transformers peft bitsandbytes accelerate datasets
```

Now configure quantization.

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
```

Create configuration:

```python
quantization_config = BitsAndBytesConfig(

    # Load base model in 4-bit
    load_in_4bit=True,

    # NF4 quantization
    bnb_4bit_quant_type="nf4",

    # Compute dtype
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Nested / double quantization
    bnb_4bit_use_double_quant=True
)
```

Load the model:

```python
MODEL_NAME = "your-model-name"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
```

Now:

```text
Model weights
     ↓
4-bit quantized representation
     ↓
Loaded into GPU
```

---

# 16. Prepare the quantized model for training

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

This prepares the model for k-bit PEFT training.

Then add LoRA.

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

For a LLaMA-style architecture:

```python
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
```

Apply it:

```python
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Now:

```text
Quantized Base Model (Frozen)
            +
        LoRA A/B
            ↓
       Trainable
```

---

# 17. Complete QLoRA training example

Below is the typical architecture.

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
```

---

## Step 1: Dataset

```python
data = {
    "instruction": [
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue."
    ],

    "input": [
        "I was charged twice.",
        "I cannot log in.",
        "My application crashes."
    ],

    "output": [
        "billing",
        "account_access",
        "technical"
    ]
}

dataset = Dataset.from_dict(data)
```

---

## Step 2: Quantization configuration

```python
MODEL_NAME = "your-llama-style-model"


quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

---

## Step 3: Load tokenizer and quantized model

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

---

## Step 4: Prepare for k-bit training

```python
model = prepare_model_for_kbit_training(
    model
)
```

---

## Step 5: Add LoRA

```python
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


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()
```

---

## Step 6: Tokenize

```python
MAX_LENGTH = 512


def tokenize_example(example):

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    response = (
        example["output"]
        + tokenizer.eos_token
    )

    full_text = prompt + response

    tokens = tokenizer(

        full_text,

        truncation=True,

        max_length=MAX_LENGTH
    )


    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )


    labels = tokens[
        "input_ids"
    ].copy()


    prompt_length = len(
        prompt_tokens[
            "input_ids"
        ]
    )


    # Mask prompt tokens
    labels[:prompt_length] = (
        [-100] * prompt_length
    )


    tokens["labels"] = labels

    return tokens
```

Apply:

```python
tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)
```

---

## Step 7: Training

```python
training_args = TrainingArguments(

    output_dir="./qlora_output",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    bf16=True,

    report_to="none"
)
```

Why:

```python
gradient_accumulation_steps=4
```

Because if GPU memory only allows:

```text
batch size = 1
```

we can accumulate gradients.

Conceptually:

```text
Batch 1
   ↓
Store gradients

Batch 2
   ↓
Add gradients

Batch 3
   ↓
Add gradients

Batch 4
   ↓
optimizer.step()
```

This approximates a larger effective batch size.

Create the trainer:

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)
```

Train:

```python
trainer.train()
```

Save:

```python
model.save_pretrained(
    "./qlora_adapter"
)

tokenizer.save_pretrained(
    "./qlora_adapter"
)
```

Usually, this saves the adapter rather than a full copy of the original base model.

---

# 18. What happens internally during QLoRA?

Suppose:

```text
User Input
    ↓
"I was charged twice"
```

### Step 1: Tokenize

```text
Text
 ↓
Token IDs
```

### Step 2: Forward pass

```text
Token IDs
    ↓
Quantized Base Model
    ↓
LoRA Adapters
    ↓
Prediction
```

### Step 3: Calculate loss

```text
Expected:
billing

Predicted:
technical
```

```text
Cross Entropy Loss
        ↓
      Error
```

### Step 4: Backpropagation

```text
Loss
 ↓
Backpropagation
 ↓

Quantized Base Model
        ❄️
      Frozen

LoRA A/B
        ✓
    Updated
```

---

# 19. LoRA vs QLoRA

| Feature                         | LoRA                               | QLoRA       |
| ------------------------------- | ---------------------------------- | ----------- |
| Base model                      | Usually loaded in normal precision | Quantized   |
| Base model trainable            | No                                 | No          |
| Adapter                         | LoRA                               | LoRA        |
| Memory usage                    | Low                                | Lower       |
| Quantization                    | No                                 | Yes         |
| Typical base weight storage     | Higher precision                   | Often 4-bit |
| Training complexity             | Simpler                            | More setup  |
| Suitable for limited GPU memory | Good                               | Excellent   |

---

# 20. Full fine-tuning vs LoRA vs QLoRA

```text
FULL FINE-TUNING

Base Model
    │
    ▼
Update everything
    │
    ▼
Highest memory/cost


LoRA

Base Model
    │
    ├── Frozen ❄️
    │
    └── LoRA adapters
            │
            ▼
          Update


QLoRA

Base Model
    │
    ▼
4-bit quantization
    │
    ▼
Frozen ❄️
    +
LoRA adapters
    │
    ▼
Update
```

|                     | Full Fine-Tuning | LoRA     | QLoRA           |
| ------------------- | ---------------- | -------- | --------------- |
| Update base weights | Yes              | No       | No              |
| Add adapters        | No               | Yes      | Yes             |
| Quantize base model | Optional         | Optional | Core idea       |
| Memory              | Highest          | Lower    | Lowest of these |
| Training cost       | Highest          | Lower    | Lower           |
| Flexibility         | High             | High     | High            |

---

# 21. When should you use each?

## Use Full Fine-Tuning when

```text
You have:
✓ Huge dataset
✓ Large GPU infrastructure
✓ Need maximum model adaptation
```

---

## Use LoRA when

```text
You have:
✓ A pre-trained model
✓ Moderate GPU resources
✓ Domain/task adaptation
✓ Multiple specialized adapters
```

---

## Use QLoRA when

```text
You have:
✓ Limited GPU memory
✓ Large model
✓ Need affordable fine-tuning
✓ Cannot perform full fine-tuning
```

---

# 22. Real-world example

Imagine your company has:

```text
One Base LLM
```

Different teams need different behavior:

```text
                    Base Model
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Finance        Legal         Support
       LoRA           LoRA          LoRA
      Adapter        Adapter        Adapter
```

You do not need:

```text
Finance: 70B model
Legal:   70B model
Support: 70B model
```

Instead:

```text
One 70B Base Model
+
Small Finance Adapter
+
Small Legal Adapter
+
Small Support Adapter
```

This can make model specialization and versioning much more manageable, depending on the serving system.

---

# 23. Most important interview answer

### What is LoRA?

> **"LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning technique where the original model weights are frozen and the required weight update is represented using small low-rank trainable matrices. Instead of updating a large weight matrix W directly, LoRA learns a low-rank update ΔW, commonly represented as B × A. This significantly reduces the number of trainable parameters, GPU memory, optimizer state, and checkpoint size."**

### What is QLoRA?

> **"QLoRA combines quantization with LoRA. The large base model is loaded in a low-bit quantized representation, commonly 4-bit, and kept frozen, while LoRA adapters are trained for the target task. This reduces memory consumption further and makes fine-tuning larger models possible on more limited hardware. In practice, QLoRA commonly uses 4-bit NF4 quantization, optional double quantization, and higher-precision computation for the training operations."**

## The easiest way to remember

```text
LoRA
=
Frozen Base Model
+
Trainable Low-Rank Adapters


QLoRA
=
Quantized Frozen Base Model
+
Trainable Low-Rank Adapters
```

For an AI/LLM interview, remember this flow:

```text
Dataset
   ↓
Tokenizer
   ↓
Frozen Base LLM
   +
LoRA A/B adapters
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Update adapters only
```
