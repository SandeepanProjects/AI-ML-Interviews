# How would you fine-tune Llama using Hugging Face?

The most practical answer today is usually:

```text
Llama base/instruct model
        +
Hugging Face Transformers
        +
TRL SFTTrainer
        +
PEFT / LoRA
        ↓
Fine-tuned Llama adapter
```

For most real-world projects, I would **not start with full fine-tuning** of Llama. I would typically use **LoRA**, and if GPU memory is limited, **QLoRA**.

Hugging Face TRL supports supervised fine-tuning with `SFTTrainer`, including PEFT integration, and its current documentation recommends adapter-based training through `peft_config`. ([Hugging Face][1])

---

# 1. What are we building?

Let's fine-tune a Llama-style instruct model to become a customer-support assistant.

Architecture:

```text
                 Training Data
                      │
                      ▼
              Format as conversation
                      │
                      ▼
                Llama Tokenizer
                      │
                      ▼
                Llama Base Model
                      │
                ┌─────┴─────┐
                │   LoRA    │
                │ Adapters  │
                └─────┬─────┘
                      │
                      ▼
                  SFTTrainer
                      │
                      ▼
               Fine-tuned Model
```

---

# 2. Install required libraries

```bash
pip install torch transformers datasets accelerate
pip install trl peft bitsandbytes
```

For current PEFT + TRL workflows, Hugging Face documents `trl`, `peft`, and optionally `bitsandbytes` for quantized training. ([Hugging Face][2])

Check versions:

```python
import transformers
import trl
import peft
import torch

print(transformers.__version__)
print(trl.__version__)
print(peft.__version__)
print(torch.__version__)
```

---

# 3. Choose a Llama model

For example:

```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
```

You can replace this with another Llama checkpoint that you have permission to access.

Important: Meta Llama models may require accepting the model license/access terms on the model repository before downloading.

Conceptually:

```text
Small model
    ↓
Lower GPU requirements
    ↓
Good for learning


Large model
    ↓
Higher quality
    ↓
More GPU memory
```

For learning:

```text
1B–3B
```

is easier.

For production:

```text
8B+
```

is common depending on quality, latency, and infrastructure requirements.

---

# 4. Create the fine-tuning dataset

A conversational dataset can look like this:

```python
from datasets import Dataset


data = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant."
                )
            },
            {
                "role": "user",
                "content": (
                    "My payment failed. What should I do?"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "I'm sorry your payment failed. "
                    "Please verify your card details and available balance. "
                    "If the problem continues, please contact support "
                    "with the error message."
                )
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant."
                )
            },
            {
                "role": "user",
                "content": (
                    "How do I reset my password?"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Click 'Forgot Password' on the login page "
                    "and follow the password reset instructions."
                )
            }
        ]
    }
]

dataset = Dataset.from_list(data)

print(dataset)
```

Output conceptually:

```text
Dataset({
    features: ['messages'],
    num_rows: 2
})
```

---

# 5. Why use `messages` format?

Modern chat models are trained with a structured conversation format:

```text
System
   ↓
User
   ↓
Assistant
```

Example:

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG."
        },
        {
            "role": "assistant",
            "content": "RAG retrieves relevant information..."
        }
    ]
}
```

The model tokenizer can convert this into the correct chat template.

For Llama-style chat models, this is preferable to manually inventing special tokens.

---

# 6. Load the tokenizer

```python
from transformers import AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Set the padding token if needed:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Why?

During batch training:

```text
Example 1:

Hello
```

```text
Example 2:

Hello, how can I reset my password?
```

Sequences have different lengths.

They need padding:

```text
Hello <PAD> <PAD> <PAD>

Hello how can I reset my password
```

---

# 7. Apply the chat template

Let's see what the conversation becomes.

```python
example = dataset[0]["messages"]

formatted_text = tokenizer.apply_chat_template(
    example,
    tokenize=False
)

print(formatted_text)
```

Conceptually:

```text
<system>
You are a helpful customer support assistant.

<user>
My payment failed. What should I do?

<assistant>
I'm sorry your payment failed...
```

The exact special tokens depend on the tokenizer/model.

This is important:

> Don't manually guess Llama's special tokens. Use the model's chat template.

---

# 8. Method 1: Full fine-tuning

Let's first understand full fine-tuning.

Load the model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Check trainable parameters:

```python
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Trainable:", trainable_parameters)
print("Total:", total_parameters)
```

With full fine-tuning:

```text
Trainable parameters
        =
Total model parameters
```

For a 3B model:

```text
3,000,000,000 parameters

        ↓

All are trainable
```

This is expensive.

---

# 9. Fine-tune using SFTTrainer

```python
from trl import (
    SFTTrainer,
    SFTConfig
)
```

Configuration:

```python
training_args = SFTConfig(

    output_dir="./llama-sft",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    logging_steps=10,

    save_steps=100,

    eval_strategy="no",

    bf16=True,

    max_length=1024
)
```

Create the trainer:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./llama-sft-final"
)

tokenizer.save_pretrained(
    "./llama-sft-final"
)
```

This is the basic full SFT workflow. `SFTTrainer` supports causal language models and current conversational/instruction dataset workflows. ([Hugging Face][1])

---

# 10. But in practice, I recommend LoRA

For a large Llama model:

```text
Full Fine-Tuning:

Llama Weights
████████████████████████
████████████████████████
████████████████████████

Everything updates
```

With LoRA:

```text
Llama Base Model
████████████████████████
        FROZEN

        +

LoRA
██
██

Only adapters train
```

PEFT is designed to fine-tune a small number of additional/trainable parameters instead of updating the full pretrained model. ([Hugging Face][3])

---

# 11. Fine-tuning Llama using LoRA

Create the LoRA configuration:

```python
from peft import LoraConfig


peft_config = LoraConfig(

    # Rank of LoRA matrices
    r=16,

    # Scaling parameter
    lora_alpha=32,

    # Regularization
    lora_dropout=0.05,

    # Llama projection layers
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

Current TRL examples show `LoraConfig` passed directly to `SFTTrainer` through `peft_config`, with target modules such as attention projections configurable explicitly. ([Hugging Face][2])

---

# 12. Complete LoRA fine-tuning code

Here is a clean implementation.

## `train_lora.py`

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig
)

from trl import (
    SFTConfig,
    SFTTrainer
)


# ==========================================
# 1. Configuration
# ==========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

OUTPUT_DIR = "./llama-support-lora"


# ==========================================
# 2. Dataset
# ==========================================

data = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional customer support assistant."
                )
            },
            {
                "role": "user",
                "content": (
                    "My payment failed."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "I'm sorry your payment failed. "
                    "Please check your payment details and "
                    "available balance. If the issue continues, "
                    "contact support with the error message."
                )
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional customer support assistant."
                )
            },
            {
                "role": "user",
                "content": (
                    "How do I reset my password?"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Click 'Forgot Password' on the login page "
                    "and follow the instructions sent to your "
                    "registered email address."
                )
            }
        ]
    }
]

dataset = Dataset.from_list(
    data
)


# ==========================================
# 3. Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 4. Model
# ==========================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# ==========================================
# 5. LoRA Configuration
# ==========================================

peft_config = LoraConfig(

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


# ==========================================
# 6. Training Configuration
# ==========================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    bf16=True,

    max_length=1024,

    packing=False,

    report_to="none"
)


# ==========================================
# 7. Trainer
# ==========================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=peft_config
)


# ==========================================
# 8. Train
# ==========================================

trainer.train()


# ==========================================
# 9. Save Adapter
# ==========================================

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)
```

The training flow is:

```text
Dataset
   ↓
Tokenizer
   ↓
Chat Template
   ↓
Token IDs
   ↓
Llama
   ↓
LoRA Adapters
   ↓
Loss
   ↓
Backpropagation
   ↓
Only LoRA parameters update
```

---

# 13. Check how many parameters are actually trainable

This is a useful interview demonstration.

```python
def print_trainable_parameters(model):

    trainable_params = 0
    total_params = 0

    for parameter in model.parameters():

        total_params += parameter.numel()

        if parameter.requires_grad:

            trainable_params += parameter.numel()

    percentage = (
        100
        * trainable_params
        / total_params
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )
```

After applying LoRA:

```python
print_trainable_parameters(
    trainer.model
)
```

Conceptually:

```text
Full Fine-Tuning:

Trainable: 3,000,000,000
Total:     3,000,000,000

100% trainable
```

LoRA:

```text
Trainable: 10,000,000
Total:     3,000,000,000

0.3% trainable
```

The exact number depends on rank and target modules.

---

# 14. QLoRA: if GPU memory is limited

Now suppose you don't have enough GPU memory for the base model in BF16.

Use:

```text
4-bit quantized Llama
          +
BF16 LoRA adapters
```

Architecture:

```text
Llama Model
    │
    ▼
4-bit Quantization
    │
    ▼
Frozen Base Model
    │
    +
    │
LoRA Adapters
BF16 / training precision
    │
    ▼
Fine-tuned Adapter
```

---

# 15. QLoRA code

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
```

Configure 4-bit quantization:

```python
bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

Load the model:

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)
```

Prepare the model for k-bit training:

```python
from peft import prepare_model_for_kbit_training


model = prepare_model_for_kbit_training(
    model
)
```

Then use the same LoRA config:

```python
peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)
```

Then train:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=peft_config
)

trainer.train()
```

TRL's PEFT integration documents QLoRA/quantized-base workflows and notes that `bitsandbytes` is needed for 4-bit or 8-bit loading. ([Hugging Face][2])

---

# 16. Complete QLoRA training code

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training
)

from trl import (
    SFTTrainer,
    SFTConfig
)


# =====================================
# CONFIG
# =====================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

OUTPUT_DIR = "./llama-qlora-output"


# =====================================
# DATASET
# =====================================

dataset = Dataset.from_list([
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant."
                )
            },
            {
                "role": "user",
                "content": (
                    "My order is delayed."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "I'm sorry your order is delayed. "
                    "Please provide your order ID and I can "
                    "help check its current shipping status."
                )
            }
        ]
    }
])


# =====================================
# TOKENIZER
# =====================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# =====================================
# 4-BIT CONFIGURATION
# =====================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# =====================================
# LOAD QUANTIZED MODEL
# =====================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)


# =====================================
# PREPARE FOR QLORA
# =====================================

model = prepare_model_for_kbit_training(
    model
)


# =====================================
# LORA
# =====================================

peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)


# =====================================
# TRAINING CONFIG
# =====================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    save_strategy="epoch",

    max_length=1024,

    bf16=True,

    report_to="none"
)


# =====================================
# TRAINER
# =====================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=peft_config
)


# =====================================
# TRAIN
# =====================================

trainer.train()


# =====================================
# SAVE
# =====================================

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)
```

---

# 17. What exactly gets saved?

With LoRA:

```python
trainer.save_model(
    "./llama-lora"
)
```

Usually you save primarily:

```text
llama-lora/
│
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

You do **not necessarily save another complete copy of the multi-billion-parameter base model**.

This is one major benefit of adapters.

```text
Base Llama Model
       +
Customer Support Adapter
       +
Finance Adapter
       +
Coding Adapter
```

You can maintain multiple lightweight adapters.

---

# 18. Load the fine-tuned LoRA model

First load the original base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)
```

Load the adapter:

```python
model = PeftModel.from_pretrained(

    base_model,

    "./llama-support-lora"
)
```

Now the architecture is:

```text
Base Llama
     +
LoRA Adapter
     =
Fine-tuned Model
```

---

# 19. Run inference

Create messages:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful customer support assistant."
        )
    },
    {
        "role": "user",
        "content": (
            "My payment failed. What should I do?"
        )
    }
]
```

Apply the chat template:

```python
inputs = tokenizer.apply_chat_template(

    messages,

    add_generation_prompt=True,

    tokenize=True,

    return_tensors="pt",

    return_dict=True
)
```

Move inputs to the model device:

```python
inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=200,

        temperature=0.7,

        do_sample=True
    )
```

Decode:

```python
input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[
    0,
    input_length:
]

response = tokenizer.decode(

    generated_tokens,

    skip_special_tokens=True
)

print(response)
```

---

# 20. Training vs inference architecture

## Training

```text
                         Dataset
                            │
                            ▼
                      Llama Tokenizer
                            │
                            ▼
                       Llama Model
                    (mostly frozen)
                            │
                            ▼
                       LoRA Layers
                    (trainable ✓)
                            │
                            ▼
                           Loss
                            │
                            ▼
                     Backpropagation
                            │
                            ▼
                   Update LoRA only
```

## Inference

```text
             User Question
                   │
                   ▼
                Tokenizer
                   │
                   ▼
          Base Llama + LoRA
                   │
                   ▼
              Generated Answer
```

---

# 21. Train/validation split

Don't train on everything.

```python
dataset = dataset.train_test_split(

    test_size=0.1,

    seed=42
)

train_dataset = dataset["train"]

eval_dataset = dataset["test"]
```

Then:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    peft_config=peft_config
)
```

Set evaluation:

```python
training_args = SFTConfig(

    output_dir="./output",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    eval_strategy="steps",

    eval_steps=50,

    save_steps=50,

    learning_rate=2e-4
)
```

Monitor:

```text
Training Loss ↓

Validation Loss ↓
```

But if:

```text
Training Loss ↓↓↓

Validation Loss ↑↑
```

you may have:

```text
Overfitting
```

---

# 22. Add gradient checkpointing

For larger Llama models:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

Why?

Normally:

```text
Forward Pass
     ↓
Store activations
     ↓
Backward Pass
```

Gradient checkpointing:

```text
Forward Pass
     ↓
Store fewer activations
     ↓
Recompute some activations later
```

Result:

```text
Less GPU memory
        ↓
More computation
```

This is often useful when fine-tuning larger models.

---

# 23. Choosing hyperparameters

A reasonable starting point for LoRA SFT:

```python
training_args = SFTConfig(

    output_dir="./output",

    num_train_epochs=1,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    max_length=2048,

    warmup_ratio=0.03,

    logging_steps=10,

    save_strategy="steps",

    save_steps=100
)
```

Typical starting points:

| Parameter             | Starting range             |
| --------------------- | -------------------------- |
| LoRA rank             | 8–32                       |
| LoRA alpha            | 16–64                      |
| LoRA dropout          | 0.0–0.1                    |
| LoRA learning rate    | around 1e-4 to 2e-4        |
| Epochs                | 1–3                        |
| Batch size            | Depends on GPU             |
| Gradient accumulation | Increase if memory limited |

These are starting points, not universal rules. Current TRL guidance notes that adapter training commonly uses a higher learning rate than full-model training because only the new adapter parameters are being learned. ([Hugging Face][1])

---

# 24. Which method should you choose?

## Full fine-tuning

```text
Use when:
- You have large GPU infrastructure
- Huge high-quality dataset
- Need deep domain adaptation
- Have a strong reason to update all weights
```

```text
Cost: High
Memory: High
```

---

## LoRA

```text
Use when:
- Standard domain adaptation
- Instruction tuning
- Customer support
- SQL generation
- Coding style
- Company terminology
```

```text
Cost: Moderate
Memory: Lower
```

---

## QLoRA

```text
Use when:
- GPU memory is limited
- Fine-tuning larger models
- You want cost-efficient experimentation
```

```text
Cost: Lower
Memory: Lowest of these three
```

---

# 25. Production project structure

For a proper fine-tuning project:

```text
llama-finetuning/
│
├── data/
│   ├── raw/
│   │   └── support.jsonl
│   │
│   └── processed/
│       ├── train.jsonl
│       └── validation.jsonl
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── configs/
│   ├── lora.yaml
│   └── qlora.yaml
│
├── tests/
│   ├── test_dataset.py
│   └── test_inference.py
│
├── requirements.txt
└── README.md
```

A production workflow would be:

```text
Raw Data
   │
   ▼
Cleaning
   │
   ▼
Deduplication
   │
   ▼
Train / Validation Split
   │
   ▼
Chat Template Formatting
   │
   ▼
Tokenization
   │
   ▼
LoRA / QLoRA Training
   │
   ▼
Evaluation
   │
   ├── Automated Metrics
   ├── Domain Tests
   └── Human Evaluation
   │
   ▼
Model Registry
   │
   ▼
Deployment
```

---

# 26. Interview answer

A strong interview answer would be:

> **To fine-tune Llama using Hugging Face, I would first choose the appropriate Llama checkpoint and create a high-quality instruction or conversational dataset. I would load the tokenizer and model using Transformers and format conversations using the tokenizer's chat template.**
>
> **For most production use cases, I would use PEFT with LoRA instead of full fine-tuning, targeting attention projection layers such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`. I would use TRL's `SFTTrainer` to perform supervised fine-tuning.**
>
> **If GPU memory is constrained, I would use QLoRA by loading the frozen base model in 4-bit precision with bitsandbytes and training LoRA adapters on top. I would monitor training and validation loss, evaluate on domain-specific test cases, save the adapter separately, and load the adapter with the base Llama model during inference.**

---

# The key code pattern to remember

```python
from transformers import AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

dataset = Dataset.from_list([
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain RAG."
            },
            {
                "role": "assistant",
                "content": (
                    "RAG retrieves relevant information "
                    "and provides it as context to an LLM."
                )
            }
        ]
    }
])


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type="CAUSAL_LM"
)


training_args = SFTConfig(

    output_dir="./llama-output",

    num_train_epochs=3,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=1024
)


trainer = SFTTrainer(

    model=MODEL_NAME,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config
)


trainer.train()

trainer.save_model(
    "./llama-output"
)
```

## Final mental model

```text
Want to teach a Llama model?

1. Collect high-quality examples
          ↓
2. Format as instruction/chat data
          ↓
3. Load Llama + tokenizer
          ↓
4. Add LoRA adapters
          ↓
5. Train using SFTTrainer
          ↓
6. Evaluate validation data
          ↓
7. Save adapter
          ↓
8. Deploy Base Model + Adapter
```

This is the approach I would recommend learning first because it maps well to real-world AI engineering interviews and projects.

[1]: https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com "SFT Trainer · Hugging Face"
[2]: https://huggingface.co/docs/trl/peft_integration?utm_source=chatgpt.com "PEFT Integration · Hugging Face"
[3]: https://huggingface.co/docs/peft/index?utm_source=chatgpt.com "PEFT · Hugging Face"
