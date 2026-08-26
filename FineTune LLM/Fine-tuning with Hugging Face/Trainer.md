# What is `Trainer` vs `SFTTrainer`?

Both are Hugging Face training abstractions, but they solve slightly different problems.

The easiest way to remember:

> **`Trainer` = general-purpose model training framework**
> **`SFTTrainer` = specialized trainer for supervised fine-tuning of LLMs**

---

# 1. Why do we need a Trainer?

Without a trainer, you manually write the training loop:

```text
Dataset
   ↓
DataLoader
   ↓
Forward Pass
   ↓
Calculate Loss
   ↓
Backward Pass
   ↓
Optimizer Step
   ↓
Scheduler Step
   ↓
Repeat
```

For example:

```python
for epoch in range(num_epochs):

    for batch in dataloader:

        # 1. Forward pass
        outputs = model(
            **batch
        )

        # 2. Get loss
        loss = outputs.loss

        # 3. Clear old gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # 6. Update learning rate
        scheduler.step()
```

This works, but production training requires much more:

* batching
* padding
* gradient accumulation
* mixed precision
* checkpoints
* logging
* evaluation
* distributed training
* multi-GPU
* gradient clipping
* learning-rate scheduling

`Trainer` helps manage these.

---

# 2. What is Hugging Face `Trainer`?

`Trainer` is a general-purpose training API from Transformers.

Import:

```python
from transformers import Trainer
```

Conceptually:

```text
                 Trainer
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
   Dataset       Model        Optimizer
      │             │             │
      └─────────────┼─────────────┘
                    ▼
              Training Loop
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Loss       Logging    Checkpoints
        │
        ▼
 Backpropagation
```

The official [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/trainer?utm_source=chatgpt.com) describes it as a complete training and evaluation loop for PyTorch models.

---

# 3. Example: `Trainer` for classification

Let's train a sentiment classifier.

Dataset:

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": [
        "This product is excellent",
        "This product is terrible",
        "I really like this",
        "I hate this product"
    ],
    "label": [
        1,
        0,
        1,
        0
    ]
})
```

---

## Step 1: Split the dataset

```python
dataset = dataset.train_test_split(
    test_size=0.25,
    seed=42
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]
```

---

# 4. Load tokenizer

```python
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

---

# 5. Tokenize data

```python
def tokenize_function(example):

    return tokenizer(

        example["text"],

        truncation=True,

        padding="max_length",

        max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function,

    batched=True
)
```

Now the data contains:

```text
text
label
input_ids
attention_mask
```

The model cannot directly understand:

```text
"This product is excellent"
```

It needs:

```text
input_ids
```

Example:

```text
"This product is excellent"

       ↓ Tokenizer

[101, 2023, 4031, 2003, 6581, 102]
```

---

# 6. Load model

```python
from transformers import (
    AutoModelForSequenceClassification
)

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=2
)
```

Architecture:

```text
Input Text
    ↓
Tokenizer
    ↓
DistilBERT
    ↓
Classification Head
    ↓
Positive / Negative
```

---

# 7. Configure training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(

    output_dir="./sentiment-model",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    logging_steps=10,

    save_strategy="epoch",

    eval_strategy="epoch",

    report_to="none"
)
```

---

# 8. Create `Trainer`

```python
from transformers import Trainer

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer
)
```

Train:

```python
trainer.train()
```

That's it.

`Trainer` internally performs something similar to:

```python
for epoch in range(num_epochs):

    for batch in dataloader:

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
```

But with many production features added.

---

# 9. What does `Trainer` handle?

```text
Trainer
│
├── Training loop
├── DataLoader
├── Optimizer
├── Learning-rate scheduler
├── Backpropagation
├── Gradient accumulation
├── Gradient clipping
├── Mixed precision
├── Evaluation
├── Checkpoints
├── Logging
└── Distributed training
```

You can think of:

```text
Trainer = General training engine
```

---

# 10. What is `SFTTrainer`?

`SFTTrainer` means:

> **Supervised Fine-Tuning Trainer**

It comes from TRL.

```python
from trl import SFTTrainer
```

It is designed specifically for LLM fine-tuning.

Conceptually:

```text
                    SFTTrainer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        LLM          Dataset         Tokenizer
          │              │
          ▼              ▼
    Causal LM       Instruction
       Loss          Formatting
          │              │
          └──────────────┘
                  │
                  ▼
              Fine-tuning
```

The [Hugging Face TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com) documents it as a trainer specialized for supervised fine-tuning of language models.

---

# 11. Why not just use `Trainer` for LLM fine-tuning?

You can.

But with `Trainer`, you often need to manually handle:

```text
Raw Dataset
     ↓
Prompt formatting
     ↓
Chat template
     ↓
Tokenization
     ↓
Labels
     ↓
Padding
     ↓
Data collator
     ↓
Trainer
```

With `SFTTrainer`, much of the LLM-specific workflow is easier.

For example:

```text
Conversation Dataset
        ↓
SFTTrainer
        ↓
Format conversation
        ↓
Tokenization
        ↓
Causal language-model loss
        ↓
Fine-tuning
```

---

# 12. `Trainer` with an LLM

Suppose you have:

```python
dataset = [
    {
        "instruction": "Explain RAG",

        "response": (
            "RAG retrieves relevant documents "
            "and provides them to the LLM."
        )
    }
]
```

With plain `Trainer`, you need to format it.

```python
def format_example(example):

    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )
```

Then tokenize:

```python
def tokenize(example):

    text = format_example(
        example
    )

    result = tokenizer(

        text,

        truncation=True,

        max_length=1024
    )

    result["labels"] = (
        result["input_ids"].copy()
    )

    return result
```

For causal language modeling:

```text
input_ids
     =
labels
```

Conceptually:

```text
Input:

Explain RAG.

RAG retrieves information.
```

The model learns:

```text
Given previous tokens
        ↓
Predict next token
```

Then create a data collator and `Trainer`.

This is more manual.

---

# 13. The same thing with `SFTTrainer`

Your dataset can be:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "instruction": "Explain RAG",

        "response": (
            "RAG retrieves relevant information "
            "and provides it to the LLM as context."
        )
    }
])
```

Or conversational:

```python
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
                    "RAG retrieves relevant documents "
                    "and provides them as context to an LLM."
                )
            }
        ]
    }
])
```

Then:

```python
from trl import SFTTrainer

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    processing_class=tokenizer,

    args=training_args
)

trainer.train()
```

Much cleaner for LLM SFT.

---

# 14. Full example: Fine-tuning Llama with `SFTTrainer`

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from trl import (
    SFTConfig,
    SFTTrainer
)


# ======================================
# Configuration
# ======================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ======================================
# Dataset
# ======================================

dataset = Dataset.from_list([

    {
        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a helpful AI assistant."
                )
            },

            {
                "role": "user",

                "content": (
                    "Explain what RAG is."
                )
            },

            {
                "role": "assistant",

                "content": (
                    "RAG stands for Retrieval-Augmented Generation. "
                    "It retrieves relevant information and gives it "
                    "to an LLM as additional context."
                )
            }
        ]
    },

    {
        "messages": [

            {
                "role": "user",

                "content": (
                    "Explain LoRA."
                )
            },

            {
                "role": "assistant",

                "content": (
                    "LoRA is a parameter-efficient fine-tuning "
                    "technique that trains small low-rank matrices "
                    "instead of updating all model parameters."
                )
            }
        ]
    }
])


# ======================================
# Tokenizer
# ======================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# ======================================
# Model
# ======================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# ======================================
# Training Config
# ======================================

training_args = SFTConfig(

    output_dir="./llama-sft-output",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-5,

    logging_steps=10,

    save_strategy="epoch",

    bf16=True,

    max_length=1024,

    report_to="none"
)


# ======================================
# SFT Trainer
# ======================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer
)


# ======================================
# Train
# ======================================

trainer.train()


# ======================================
# Save
# ======================================

trainer.save_model(
    "./llama-sft-final"
)
```

---

# 15. What happens internally in `SFTTrainer`?

Your input:

```python
{
    "messages": [

        {
            "role": "user",
            "content": "Explain RAG"
        },

        {
            "role": "assistant",
            "content": "RAG retrieves documents..."
        }
    ]
}
```

Internally, conceptually:

```text
Messages
   │
   ▼
Chat Template
   │
   ▼

<user>
Explain RAG

<assistant>
RAG retrieves documents...

   │
   ▼
Tokenizer
   │
   ▼

input_ids
attention_mask
labels

   │
   ▼
Llama
   │
   ▼
Predicted Tokens
   │
   ▼
Cross Entropy Loss
   │
   ▼
Backpropagation
```

---

# 16. How does `SFTTrainer` calculate loss?

The LLM predicts the next token.

Suppose:

```text
Input:

RAG retrieves relevant
```

Target:

```text
documents
```

Then:

```text
RAG
 ↓
retrieve

retrieves
 ↓
relevant

relevant
 ↓
documents
```

Mathematically:

$$
Loss =
-\sum_{t=1}^{T}
\log P(
y_t
|
y_{<t}
)
$$

Simplified code:

```python
import torch.nn.functional as F


def causal_lm_loss(
    logits,
    labels
):

    # Remove last prediction
    shifted_logits = logits[:, :-1]

    # Remove first label
    shifted_labels = labels[:, 1:]

    loss = F.cross_entropy(

        shifted_logits.reshape(
            -1,
            shifted_logits.size(-1)
        ),

        shifted_labels.reshape(
            -1
        ),

        ignore_index=-100
    )

    return loss
```

---

# 17. Important: Assistant-only loss

For conversational fine-tuning:

```text
User:
What is RAG?

Assistant:
RAG retrieves relevant information.
```

Do we want the model to learn to predict:

```text
User:
What is RAG?
```

Usually, the more focused objective is:

```text
Only optimize the assistant answer
```

Conceptually:

```text
User tokens:

What is RAG?
^^^^^^^^^^^^
Ignored for loss


Assistant tokens:

RAG retrieves information
^^^^^^^^^^^^^^^^^^^^^^^^^
Used for loss
```

Labels can look like:

```python
labels = [
    -100,
    -100,
    -100,
    500,
    2100,
    8721
]
```

Where:

```text
-100
```

means:

```text
Ignore this token when calculating loss
```

This is often called **assistant-only loss** or **completion-only loss**.

Current TRL `SFTTrainer` supports conversational datasets and training configurations for controlling how loss is applied, depending on dataset format and model/template support. [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com)

---

# 18. Add LoRA to `SFTTrainer`

This is where `SFTTrainer` becomes especially useful.

```python
from peft import LoraConfig


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

    task_type="CAUSAL_LM"
)
```

Then:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config
)
```

Now:

```text
Llama Base Model
       │
       ▼
Mostly Frozen
       │
       +
       │
       ▼
LoRA Adapters
       │
       ▼
SFTTrainer
       │
       ▼
Update Adapters
```

The TRL PEFT integration is specifically designed for workflows like SFT + LoRA. [TRL PEFT integration docs](https://huggingface.co/docs/trl/peft_integration?utm_source=chatgpt.com)

---

# 19. Can you use `Trainer` with LoRA?

Yes.

For example:

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)
```

Then:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset
)
```

But you still need to handle:

```text
Dataset formatting
Tokenization
Labels
Data collator
Chat template
Padding
```

So:

```text
Trainer + LoRA
      ↓
More manual control


SFTTrainer + LoRA
      ↓
More convenient for LLM SFT
```

---

# 20. Detailed comparison

| Feature              | `Trainer`                    | `SFTTrainer`               |
| -------------------- | ---------------------------- | -------------------------- |
| Library              | Transformers                 | TRL                        |
| Purpose              | General training             | LLM supervised fine-tuning |
| Classification       | Excellent                    | Not primary use            |
| Token classification | Excellent                    | Not primary use            |
| LLM SFT              | Possible                     | Specialized                |
| Chat datasets        | Manual handling often needed | Native/convenient support  |
| Chat templates       | Manual setup often needed    | Integrated workflow        |
| Instruction tuning   | Manual                       | Easier                     |
| Completion-only loss | Manual collator/labels       | Supported workflow         |
| LoRA integration     | Manual PEFT setup            | Direct PEFT integration    |
| DPO training         | No                           | Use `DPOTrainer`           |
| GRPO training        | No                           | Use `GRPOTrainer`          |

---

# 21. When should you use `Trainer`?

Use `Trainer` when you are training:

### Classification

```text
Email
 ↓
Spam / Not Spam
```

### Token classification

```text
John lives in India

John → PERSON
India → LOCATION
```

### Regression

```text
House features
     ↓
Price
```

### Custom ML tasks

```text
Input
  ↓
Transformer Model
  ↓
Custom Head
  ↓
Custom Loss
```

Example:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    compute_metrics=compute_metrics
)
```

---

# 22. When should you use `SFTTrainer`?

Use `SFTTrainer` when you want:

```text
Prompt
   ↓
LLM
   ↓
Desired Completion
```

Examples:

### Instruction tuning

```text
Explain RAG
    ↓
RAG is...
```

### Chatbot training

```text
User
 ↓
Assistant
```

### Company support bot

```text
Customer question
      ↓
Support answer
```

### SQL generation

```text
English
   ↓
SQL
```

### Code generation

```text
Requirement
    ↓
Python code
```

### Structured output

```text
Natural language
      ↓
JSON
```

---

# 23. Real-world architecture

Suppose you're building an enterprise customer-support model.

```text
                   Raw Conversations
                           │
                           ▼
                    Data Cleaning
                           │
                           ▼
                   Remove PII / Secrets
                           │
                           ▼
                  Quality Filtering
                           │
                           ▼
                    Train / Val Split
                           │
                           ▼
                     Chat Dataset
                           │
                           ▼
                    Llama Model
                           │
                           ▼
                 LoRA / QLoRA
                           │
                           ▼
                    SFTTrainer
                           │
                           ▼
                    Evaluation
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Accuracy       Safety       Style
                           │
                           ▼
                     Production
```

---

# 24. Which one should you use for your AI projects?

For the type of LLM/RAG and AI-engineering projects you're learning, I would generally use:

```text
Standard Transformer task
        ↓
      Trainer
```

```text
LLM instruction tuning
        ↓
    SFTTrainer
```

```text
Preference optimization
        ↓
    DPOTrainer
```

```text
Group-based RL optimization
        ↓
    GRPOTrainer
```

This is the TRL ecosystem:

```text
                    Base Model
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      SFTTrainer    DPOTrainer    GRPOTrainer
          │             │             │
          ▼             ▼             ▼
        SFT            DPO           RL
```

---

# 25. Interview-ready answer

> **Hugging Face `Trainer` is a general-purpose training abstraction from the Transformers library. It manages the training loop, optimization, scheduling, mixed precision, gradient accumulation, evaluation, logging, checkpointing, and distributed training. I would commonly use it for tasks such as classification, token classification, regression, or custom Transformer training.**
>
> **`SFTTrainer`, from Hugging Face TRL, is specialized for supervised fine-tuning of LLMs. It simplifies instruction and conversational dataset processing, chat templates, causal language-model training, completion/assistant-focused training workflows, and integrates naturally with PEFT methods such as LoRA and QLoRA.**
>
> **For fine-tuning Llama on instruction or chat data, I would generally choose `SFTTrainer`. For a general Transformer task such as sentiment classification, I would use `Trainer`.**

## One-line memory trick

```text
Trainer
   ↓
Train any Transformer task


SFTTrainer
   ↓
Fine-tune an LLM to generate better responses
```
