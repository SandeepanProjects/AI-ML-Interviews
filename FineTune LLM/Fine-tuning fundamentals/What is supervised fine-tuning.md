Absolutely. Let's understand **Supervised Fine-Tuning (SFT) with actual code**, from dataset → tokenization → training → inference.

I'll use the Hugging Face ecosystem because this is a common real-world/interview setup.

---

# 1. What we are going to build

We want to teach a pre-trained model this behavior:

```text
Customer message
       ↓
Fine-tuned model
       ↓
Category
```

Examples:

```text
"I was charged twice"       → billing
"I cannot log in"           → account_access
"The application crashes"   → technical
```

Our SFT process will be:

```text
Training Data
     ↓
Prompt + Expected Answer
     ↓
Tokenizer
     ↓
Pre-trained Model
     ↓
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
LoRA Adapter Update
     ↓
Fine-tuned Model
```

---

# 2. Project structure

```text
sft_example/
│
├── data/
│   ├── train.jsonl
│   └── test.jsonl
│
├── prepare_data.py
├── train.py
├── inference.py
└── requirements.txt
```

---

# 3. Install dependencies

```bash
pip install torch transformers datasets trl peft accelerate
```

For quantized QLoRA training, you may also use:

```bash
pip install bitsandbytes
```

---

# 4. Create the training dataset

Create:

```text
data/train.jsonl
```

Example:

```json
{"instruction":"Classify the customer support issue.","input":"I was charged twice for my subscription.","output":"billing"}
{"instruction":"Classify the customer support issue.","input":"I cannot log into my account.","output":"account_access"}
{"instruction":"Classify the customer support issue.","input":"The application crashes immediately after opening.","output":"technical"}
{"instruction":"Classify the customer support issue.","input":"My refund has not arrived yet.","output":"refund"}
{"instruction":"Classify the customer support issue.","input":"My payment was declined.","output":"payment_failure"}
```

This is **supervised data** because every input has a known correct output.

```text
Input                                  Label
------------------------------------------------
I was charged twice                    billing
I cannot log in                        account_access
App crashes                            technical
```

---

# 5. Format the dataset as an SFT prompt

Let's create `prepare_data.py`:

```python
from datasets import load_dataset


def format_example(example):
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    }


dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)

dataset = dataset.map(format_example)

print(dataset["train"][0]["text"])
```

The resulting training text becomes:

```text
### Instruction:
Classify the customer support issue.

### Input:
I was charged twice for my subscription.

### Response:
billing
```

This is what the model sees.

---

# 6. Load the tokenizer and model

Now create `train.py`.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
```

For GPT-2, we need a padding token:

```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

At this point:

```text
Pre-trained GPT-2
       │
       ▼
General language capability
```

We now want to specialize its behavior.

---

# 7. Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)
```

Format it:

```python
def format_example(example):
    text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""

    return {"text": text}


dataset = dataset.map(format_example)
```

---

# 8. Tokenization

The model cannot directly understand:

```text
I was charged twice
```

We convert text into token IDs.

```python
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)
```

Conceptually:

```text
"I was charged twice"
          ↓
Tokenizer
          ↓
[40, 182, 5490, 923]
```

The actual IDs depend on the tokenizer.

---

# 9. Create labels

For a causal language model, a simple training setup often uses the input token IDs as labels:

```python
def tokenize_function(example):

    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized
```

So conceptually:

```text
Input IDs:
[10, 20, 30, 40, 50]

Labels:
[10, 20, 30, 40, 50]
```

The model learns to predict the next tokens.

For chat/instruction SFT, production training often **masks the prompt tokens** and computes loss only on the assistant response. We will cover that shortly.

---

# 10. The simplest training loop

Here is the core concept without hiding anything behind a trainer.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

model.train()

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)

for epoch in range(3):

    for batch in dataloader:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # 1. Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 2. Loss
        loss = outputs.loss

        # 3. Clear previous gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update model weights
        optimizer.step()

        print("Loss:", loss.item())
```

This is the heart of SFT:

```text
Training Example
      ↓
Model Prediction
      ↓
Compare with Correct Output
      ↓
Loss
      ↓
loss.backward()
      ↓
Gradients
      ↓
optimizer.step()
      ↓
Updated Parameters
```

---

# 11. Production-friendly SFT using TRL

In real LLM projects, you normally don't manually write every part of the training loop.

TRL provides SFT utilities.

A modern approach looks like this:

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL_NAME = "gpt2"

# -----------------------
# Load tokenizer
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token


# -----------------------
# Load model
# -----------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# -----------------------
# Load dataset
# -----------------------

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)


# -----------------------
# Format examples
# -----------------------

def format_example(example):

    return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""


# -----------------------
# Training configuration
# -----------------------

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=1,
    save_steps=100,
    report_to="none",
)


# -----------------------
# SFT Trainer
# -----------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_example,
)


# -----------------------
# Train
# -----------------------

trainer.train()


# -----------------------
# Save model
# -----------------------

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

The trainer internally performs:

```text
Dataset
   ↓
Formatting
   ↓
Tokenization
   ↓
Create Batches
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Repeat
```

---

# 12. What is actually supervised here?

Look at this example:

```text
### Instruction:
Classify the customer support issue.

### Input:
I was charged twice.

### Response:
billing
```

The desired output is:

```text
billing
```

During training, the model learns that after seeing:

```text
I was charged twice
```

the probability of generating:

```text
billing
```

should increase.

Before training:

```text
Input:
I was charged twice.

Possible outputs:
- I apologize for the inconvenience
- It seems there was a duplicate charge
- billing
```

After good fine-tuning:

```text
Input:
I was charged twice.

Desired output:
billing
```

The fine-tuned model should assign a higher probability to the desired output.

---

# 13. Better SFT: calculate loss only on the answer

This is important.

Suppose your prompt is:

```text
### Instruction:
Classify the customer issue.

### Input:
I was charged twice.

### Response:
billing
```

We usually want the model to learn mainly:

```text
billing
```

We don't necessarily want to train it to reproduce the instruction.

Conceptually:

```text
Prompt tokens:
[-100, -100, -100, -100, -100]

Response tokens:
[billing_token]
```

`-100` means:

> Ignore this token when calculating the loss.

Example code:

```python
def tokenize_with_labels(example):

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    full_text = prompt + example["output"]

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=256
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=256
    )

    labels = full_tokens["input_ids"].copy()

    prompt_length = len(prompt_tokens["input_ids"])

    # Ignore prompt tokens in loss calculation
    labels[:prompt_length] = [-100] * prompt_length

    full_tokens["labels"] = labels

    return full_tokens
```

Conceptually:

```text
INPUT:

[Instruction] [Input] [Response] [billing]
      │            │        │          │
      ▼            ▼        ▼          ▼
    Ignore       Ignore   Ignore    Train
    (-100)       (-100)   (-100)    Label
```

This is closer to typical instruction SFT.

---

# 14. SFT with LoRA

Now let's look at a more practical approach.

Instead of updating all model parameters, we can use LoRA.

```python
from peft import LoraConfig, get_peft_model


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Conceptually:

```text
                Base Model
             7 Billion Parameters
                      │
                      ▼
                 Frozen Weights
                      │
              ┌───────┴───────┐
              │               │
            LoRA A          LoRA B
              │               │
              └───────┬───────┘
                      │
                      ▼
               Train Adapters
```

The core idea is:

```text
Original Weight = W

Fine-tuned Weight:

W_new = W + ΔW

LoRA approximates:

ΔW = A × B
```

Instead of training all parameters, we train smaller matrices.

---

# 15. Complete SFT + LoRA example

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer


MODEL_NAME = "gpt2"

# -----------------------
# Tokenizer
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# -----------------------
# Base model
# -----------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# -----------------------
# LoRA
# -----------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# -----------------------
# Dataset
# -----------------------

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)


# -----------------------
# Format
# -----------------------

def formatting_func(example):

    return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""


# -----------------------
# Training arguments
# -----------------------

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)


# -----------------------
# Trainer
# -----------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)


# -----------------------
# Train
# -----------------------

trainer.train()


# -----------------------
# Save adapter
# -----------------------

model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
```

---

# 16. Inference after SFT

Create `inference.py`:

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model"
)

prompt = """### Instruction:
Classify the customer support issue.

### Input:
I was charged twice on my credit card.

### Response:
"""

result = generator(
    prompt,
    max_new_tokens=20,
    do_sample=False
)

print(result[0]["generated_text"])
```

Expected conceptually:

```text
### Instruction:
Classify the customer support issue.

### Input:
I was charged twice on my credit card.

### Response:
billing
```

---

# 17. Full internal SFT flow with code mapping

Here is the complete connection:

```text
1. DATASET

{
    input: "I was charged twice",
    output: "billing"
}

          ↓

2. FORMAT

Instruction + Input + Expected Output

          ↓

3. TOKENIZE

Text → Token IDs

          ↓

4. FORWARD PASS

outputs = model(...)

          ↓

5. LOSS

loss = outputs.loss

          ↓

6. BACKPROPAGATION

loss.backward()

          ↓

7. UPDATE

optimizer.step()

          ↓

8. REPEAT

Next Batch

          ↓

9. SAVE

Fine-tuned model / adapter
```

---

# 18. Very important production point

Don't judge a fine-tuned model only by:

```text
Training loss ↓
```

You need a separate evaluation dataset.

```text
Dataset
   │
   ├── train.jsonl
   │
   ├── validation.jsonl
   │
   └── test.jsonl
```

Example:

```python
result = trainer.evaluate()

print(result)
```

For classification, evaluate:

```text
Accuracy
Precision
Recall
F1 Score
```

For LLM generation:

```text
Task success rate
Format compliance
Human evaluation
Faithfulness
Safety
Latency
Cost
```

---

# Interview answer

If asked:

> **Explain supervised fine-tuning with code.**

You can say:

> "In supervised fine-tuning, I prepare a dataset containing input and expected output pairs, usually in instruction or chat format. I tokenize the examples and pass them through a pre-trained causal language model. The model performs a forward pass and predicts output tokens. A loss function, typically token-level cross-entropy, compares the predictions with the expected output. Backpropagation calculates gradients, and an optimizer updates the trainable parameters. In practice, I often use Hugging Face Transformers with TRL's SFTTrainer and PEFT/LoRA to train only a small set of adapter parameters. I also mask prompt tokens so that the training loss focuses on the desired assistant response. Finally, I evaluate the model on a held-out test set before deployment."

### The key code to remember

```python
outputs = model(
    input_ids=input_ids,
    labels=labels
)

loss = outputs.loss

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

That is the fundamental training loop behind **SFT**.
