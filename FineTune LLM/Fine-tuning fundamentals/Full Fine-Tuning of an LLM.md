# What is Full Fine-Tuning of an LLM?

**Full fine-tuning** means taking a pre-trained LLM and continuing training on a new dataset while **updating all (or nearly all) of its model parameters**.

This is the main difference:

```text
Full Fine-Tuning:
    Base model weights → UPDATED

LoRA / PEFT:
    Base model weights → FROZEN
    Small adapter weights → UPDATED
```

---

# 1. The basic idea

Suppose you start with a pre-trained model:

```text
Pre-trained LLM
      │
      │  Already knows:
      │  - Language
      │  - Grammar
      │  - General knowledge
      │  - Reasoning patterns
      ▼
Fine-tuning dataset
      │
      ▼
Continue training
      │
      ▼
ALL model weights updated
      │
      ▼
Specialized model
```

For example:

```text
Base model:
General assistant

Fine-tuning data:
Financial reports
Financial question-answer pairs
Financial terminology
Financial instructions

Result:
Financial-specialized LLM
```

---

# 2. What exactly gets updated?

Consider a small neural network:

```text
Input
  ↓
Layer 1 → W1
  ↓
Layer 2 → W2
  ↓
Layer 3 → W3
  ↓
Output
```

During **full fine-tuning**:

```text
W1 → updated ✓
W2 → updated ✓
W3 → updated ✓
```

For a Transformer:

```text
Embedding Layers              ✓
Attention Query weights       ✓
Attention Key weights         ✓
Attention Value weights       ✓
Attention Output weights      ✓
MLP / FFN weights             ✓
LayerNorm parameters          ✓
Output / LM Head              ✓
```

In contrast, LoRA might look like:

```text
Base Transformer weights      Frozen ❄️
LoRA A/B matrices             Updated ✓
```

---

# 3. Full fine-tuning mathematically

Suppose a model has parameters:

```text
θ
```

The model receives input:

```text
x
```

and predicts:

```text
ŷ = Model(x; θ)
```

We know the expected answer:

```text
y
```

We calculate a loss:

```text
Loss = L(ŷ, y)
```

Then calculate gradients:

```text
∂Loss / ∂θ
```

And update the parameters:

```text
θ_new = θ_old - learning_rate × gradient
```

For full fine-tuning:

```text
θ1 ✓ updated
θ2 ✓ updated
θ3 ✓ updated
...
θN ✓ updated
```

That is the core definition.

---

# 4. Full fine-tuning vs pre-training

## Pre-training

Starts with:

```text
Random weights
      ↓
Massive internet/books/code data
      ↓
Train for a very long time
      ↓
General-purpose LLM
```

Example objective:

```text
The capital of France is ___
```

Model learns:

```text
Paris
```

---

## Full fine-tuning

Starts with:

```text
Already pre-trained weights
      ↓
Your specialized dataset
      ↓
Continue training
      ↓
All parameters updated
```

So:

```text
Pre-training = Learn general capabilities from scratch

Full fine-tuning = Adapt existing capabilities by updating all weights
```

---

# 5. Full fine-tuning vs LoRA

```text
FULL FINE-TUNING

Pre-trained model
       │
       ▼
All parameters trainable
       │
       ▼
Backpropagation
       │
       ▼
Update all weights
```

```text
LoRA

Pre-trained model
       │
       ├── Base weights frozen ❄️
       │
       └── LoRA adapters trainable ✓
               │
               ▼
         Update adapters only
```

| Feature                      | Full Fine-Tuning | LoRA              |
| ---------------------------- | ---------------- | ----------------- |
| Base weights                 | Updated          | Frozen            |
| Trainable parameters         | Almost/all       | Small subset      |
| GPU memory                   | High             | Lower             |
| Training cost                | High             | Lower             |
| Checkpoint                   | Full model       | Small adapter     |
| Adaptation capacity          | Maximum          | Usually very good |
| Catastrophic forgetting risk | Higher           | Generally lower   |

---

# 6. Full fine-tuning with actual code

Let's build a simple instruction fine-tuning example.

## Install packages

```bash
pip install torch transformers datasets accelerate
```

---

# 7. Create a training dataset

We'll create a small dataset for demonstration.

```python
from datasets import Dataset

data = {
    "instruction": [
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue."
    ],
    "input": [
        "I was charged twice for my subscription.",
        "I cannot log into my account.",
        "The application crashes when I open it.",
        "My refund has not arrived yet."
    ],
    "output": [
        "billing",
        "account_access",
        "technical",
        "refund"
    ]
}

dataset = Dataset.from_dict(data)

print(dataset[0])
```

Output:

```python
{
    "instruction": "Classify the customer issue.",
    "input": "I was charged twice for my subscription.",
    "output": "billing"
}
```

---

# 8. Load a pre-trained model

For demonstration, use a small model.

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
```

GPT-2 needs a padding token:

```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

At this point, check the parameters:

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
```

In **full fine-tuning**, you should see:

```text
Total parameters:     ~124M
Trainable parameters: ~124M
```

Because all model parameters are trainable.

---

# 9. Format the training examples

For causal language model training:

```python
def format_example(example):

    return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
```

Example:

```text
### Instruction:
Classify the customer issue.

### Input:
I was charged twice.

### Response:
billing
```

---

# 10. Tokenize and create labels

This is an important part.

```python
MAX_LENGTH = 256


def tokenize_example(example):

    # Prompt
    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    # Expected model answer
    response = example["output"] + tokenizer.eos_token

    # Full text given to model
    full_text = prompt + response

    # Tokenize complete sequence
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Tokenize prompt separately
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Labels start as input token IDs
    labels = tokens["input_ids"].copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # Ignore prompt tokens when calculating loss
    labels[:prompt_length] = [-100] * prompt_length

    tokens["labels"] = labels

    return tokens
```

Apply it:

```python
tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)
```

---

# 11. Why use `-100`?

Suppose the tokens are:

```text
Instruction tokens:
[10, 20, 30, 40]

Input tokens:
[50, 60, 70]

Response tokens:
[100, 101]
```

The labels become:

```text
[-100, -100, -100, -100,
 -100, -100, -100,
 100, 101]
```

So:

```text
Instruction → ignored for loss
Input       → ignored for loss
Response    → used for loss
```

This tells the model:

> Given the instruction and input, learn to generate the expected response.

---

# 12. Train using Hugging Face Trainer

```python
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
```

Training configuration:

```python
training_args = TrainingArguments(
    output_dir="./full_finetuned_model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=2e-5,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none"
)
```

Data collator:

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)
```

Create the trainer:

```python
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

This is **full fine-tuning** because we never froze the base model.

All parameters have:

```python
parameter.requires_grad == True
```

---

# 13. Complete code

Here is the complete example.

```python
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "gpt2"
MAX_LENGTH = 256


# ==================================================
# 2. DATASET
# ==================================================

data = {
    "instruction": [
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue.",
        "Classify the customer issue."
    ],

    "input": [
        "I was charged twice for my subscription.",
        "I cannot log into my account.",
        "The application crashes when I open it.",
        "My refund has not arrived yet."
    ],

    "output": [
        "billing",
        "account_access",
        "technical",
        "refund"
    ]
}

dataset = Dataset.from_dict(data)


# ==================================================
# 3. TOKENIZER
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# ==================================================
# 4. BASE MODEL
# ==================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# ==================================================
# 5. VERIFY FULL FINE-TUNING
# ==================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)


# ==================================================
# 6. TOKENIZATION
# ==================================================

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


    # Full sequence
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH
    )


    # Prompt separately
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )


    # Labels
    labels = tokens["input_ids"].copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )


    # Ignore prompt in loss
    labels[:prompt_length] = (
        [-100] * prompt_length
    )


    tokens["labels"] = labels

    return tokens


tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)


# ==================================================
# 7. DATA COLLATOR
# ==================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# ==================================================
# 8. TRAINING CONFIG
# ==================================================

training_args = TrainingArguments(

    output_dir="./full_finetuned_model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=2e-5,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none"
)


# ==================================================
# 9. TRAINER
# ==================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)


# ==================================================
# 10. TRAIN
# ==================================================

trainer.train()


# ==================================================
# 11. SAVE FULL MODEL
# ==================================================

trainer.save_model(
    "./full_finetuned_model"
)

tokenizer.save_pretrained(
    "./full_finetuned_model"
)
```

---

# 14. What happens internally during training?

Suppose we have:

```text
Input:

"I was charged twice."
```

Expected answer:

```text
billing
```

The training flow is:

```text
                    Training Example
                           │
                           ▼
              Instruction + Input + Response
                           │
                           ▼
                       Tokenizer
                           │
                           ▼
                       Token IDs
                           │
                           ▼
                    Pre-trained LLM
                           │
                           ▼
                    Forward Pass
                           │
                           ▼
                  Token Probabilities
                           │
                           ▼
                   Cross Entropy Loss
                           │
                           ▼
                    Backpropagation
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Attention          MLP           Embeddings
       Weights           Weights          Weights
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  ALL PARAMETERS UPDATED
```

---

# 15. Manual training loop

To really understand full fine-tuning, here is a simplified manual loop.

```python
import torch
from torch.optim import AdamW


# Move model to GPU if available
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

model.train()


# Optimizer receives ALL model parameters
optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)


for epoch in range(3):

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch["labels"].to(device)


        # -------------------------
        # 1. Forward pass
        # -------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        # -------------------------
        # 2. Calculate loss
        # -------------------------

        loss = outputs.loss


        # -------------------------
        # 3. Clear old gradients
        # -------------------------

        optimizer.zero_grad()


        # -------------------------
        # 4. Backpropagation
        # -------------------------

        loss.backward()


        # -------------------------
        # 5. Update ALL weights
        # -------------------------

        optimizer.step()


        print(
            f"Loss: {loss.item():.4f}"
        )
```

The most important line is:

```python
optimizer = AdamW(
    model.parameters()
)
```

Because:

```text
model.parameters()
      ↓
All trainable parameters
      ↓
Optimizer
      ↓
Updated during optimizer.step()
```

---

# 16. Compare actual LoRA code

## Full fine-tuning

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)
```

All parameters:

```text
✓ Embeddings
✓ Attention
✓ MLP
✓ LayerNorm
✓ Output layers
```

---

## LoRA

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = get_peft_model(
    model,
    lora_config
)

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=2e-4
)
```

Now:

```text
Base model parameters → Frozen ❄️
LoRA parameters       → Trainable ✓
```

This is the most important code-level difference.

---

# 17. Why does full fine-tuning require more GPU memory?

During training, memory is required for:

```text
Model weights
       +
Gradients
       +
Optimizer states
       +
Activations
```

Suppose the model has:

```text
7 billion parameters
```

With full fine-tuning:

```text
7B parameters
    ↓
Store model weights

7B parameters
    ↓
Store gradients

7B parameters
    ↓
Optimizer state
```

The exact memory depends heavily on:

* precision (`FP32`, FP16, BF16)
* optimizer
* activation memory
* sequence length
* batch size
* distributed training strategy

But the important concept is:

> Full fine-tuning needs gradients and optimizer state for the model parameters being trained, so it is much more memory-intensive than PEFT.

---

# 18. Why use full fine-tuning?

Full fine-tuning can make sense when:

### 1. You need deep model adaptation

```text
General Model
      ↓
Very different domain
      ↓
Large high-quality dataset
      ↓
Full Fine-Tuning
```

### 2. You have enough compute

For example:

```text
Multi-GPU cluster
High-memory GPUs
Distributed training
```

### 3. PEFT is not achieving the required quality

You may experiment with:

```text
Prompt Engineering
        ↓
RAG
        ↓
LoRA / QLoRA
        ↓
Full Fine-Tuning
```

You should not automatically jump to full fine-tuning.

---

# 19. Risks of full fine-tuning

## A. Catastrophic forgetting

The model may become worse at previously learned tasks.

```text
Before:
Good general knowledge

After narrow domain fine-tuning:
Excellent domain knowledge
But weaker general behavior
```

---

## B. Overfitting

Small dataset:

```text
100 examples
```

Large model:

```text
7B parameters
```

The model can memorize the dataset.

---

## C. High cost

Full fine-tuning requires more:

```text
GPU memory
GPU time
Storage
Infrastructure
```

---

# 20. Real-world example

Suppose you have a 70B model and a financial dataset.

### Option 1: Prompt engineering

```text
System Prompt:
"You are a financial assistant..."
```

No training.

Use when:

```text
Behavior change is simple.
```

---

### Option 2: RAG

```text
Question
   ↓
Retrieve current financial documents
   ↓
LLM
```

Use when:

```text
Knowledge changes frequently.
```

---

### Option 3: LoRA

```text
Base Model
+
Financial LoRA Adapter
```

Use when:

```text
You need specialized behavior
but want lower cost.
```

---

### Option 4: Full fine-tuning

```text
Base Model
       ↓
Update all weights
       ↓
Deeply specialized model
```

Use when:

```text
You have:
✓ Large high-quality dataset
✓ Enough compute
✓ Strong need for deep adaptation
```

---

# 21. Production-style training configuration

For a real model, you would likely use additional techniques:

```text
Full Fine-Tuning
      │
      ├── BF16 / FP16 mixed precision
      ├── Gradient checkpointing
      ├── Gradient accumulation
      ├── Distributed training
      ├── DeepSpeed / FSDP
      ├── Learning-rate scheduling
      ├── Evaluation
      └── Checkpointing
```

Example:

```python
training_args = TrainingArguments(
    output_dir="./model_output",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    bf16=True,

    gradient_checkpointing=True,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    report_to="none"
)
```

Conceptually:

```text
GPU allows:
Batch size = 1

Gradient accumulation = 8

Effective batch ≈ 8
```

---

# 22. Full fine-tuning vs LoRA vs QLoRA

| Feature              | Full Fine-Tuning | LoRA           | QLoRA              |
| -------------------- | ---------------- | -------------- | ------------------ |
| Base model           | Trainable        | Frozen         | Quantized + Frozen |
| Trainable parameters | All/nearly all   | Small fraction | Small fraction     |
| GPU memory           | Highest          | Lower          | Usually lowest     |
| Training cost        | Highest          | Lower          | Lower              |
| Model checkpoint     | Full model       | Adapter        | Adapter            |
| Adaptation           | Maximum capacity | Strong         | Strong             |
| Hardware requirement | High             | Moderate       | Lower              |

---

# Interview answer

> **"Full fine-tuning is the process of taking a pre-trained language model and continuing training on a task or domain-specific dataset while updating all, or nearly all, of its parameters. During training, the input passes through the model, token-level cross-entropy loss is calculated against the expected output, and backpropagation computes gradients for the entire trainable model. The optimizer then updates all model weights. Compared with LoRA or QLoRA, full fine-tuning provides greater adaptation capacity but requires significantly more GPU memory and compute because gradients and optimizer states must be maintained for the entire model."**

## Remember this:

```text
FULL FINE-TUNING

Pre-trained Model
        ↓
   All Weights
   Trainable ✓
        ↓
Forward Pass
        ↓
Loss
        ↓
Backward Pass
        ↓
ALL weights updated
```

The core difference is simply:

```python
# Full fine-tuning
for parameter in model.parameters():
    parameter.requires_grad = True
```

versus LoRA:

```python
# LoRA
Base model parameters = frozen
LoRA adapter parameters = trainable
```

For interviews, the best progression to remember is:

```text
Prompt Engineering
       ↓
RAG
       ↓
LoRA / QLoRA
       ↓
Full Fine-Tuning
```

You generally choose the **least expensive technique that solves the problem well**.
