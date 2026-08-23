# LoRA vs Full Fine-Tuning vs QLoRA

You wrote “LoRA” twice, so I assume you want:

> **Full Fine-Tuning vs LoRA vs QLoRA**

This is a very important LLM interview topic. Let's understand it from **concept → internal working → code → production decision**.

---

# 1. One-line difference

```text
Full Fine-Tuning
→ Update the entire model

LoRA
→ Freeze the base model and train small adapters

QLoRA
→ Quantize and freeze the base model, then train small LoRA adapters
```

---

# 2. Visual comparison

```text
┌─────────────────────────────────────────────────────┐
│                 FULL FINE-TUNING                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Pretrained Model                                   │
│       │                                             │
│       ▼                                             │
│  ALL PARAMETERS TRAINABLE ✓                         │
│       │                                             │
│       ▼                                             │
│  Backpropagation                                    │
│       │                                             │
│       ▼                                             │
│  ALL WEIGHTS UPDATED                                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```text
┌─────────────────────────────────────────────────────┐
│                       LoRA                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Pretrained Model                                   │
│       │                                             │
│       ▼                                             │
│  BASE WEIGHTS FROZEN ❄️                             │
│       │                                             │
│       ├─────────────┐                               │
│       │             ▼                               │
│       │       LoRA Adapters                         │
│       │       Trainable ✓                           │
│       │             │                               │
│       ▼             ▼                               │
│       └──────► Combined Output                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```text
┌─────────────────────────────────────────────────────┐
│                      QLoRA                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Pretrained Model                                   │
│       │                                             │
│       ▼                                             │
│  Quantize to 4-bit                                  │
│       │                                             │
│       ▼                                             │
│  Base Model Frozen ❄️                               │
│       │                                             │
│       ├─────────────┐                               │
│       │             ▼                               │
│       │       LoRA Adapters                         │
│       │       Trainable ✓                           │
│       │             │                               │
│       ▼             ▼                               │
│       └──────► Combined Output                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

# 3. What happens to model parameters?

Imagine a model with:

```text
7 Billion Parameters
```

## Full fine-tuning

```text
7B parameters

All 7B:
✓ Trainable
✓ Gradients
✓ Optimizer states
```

---

## LoRA

```text
7B Base Parameters
        ↓
      Frozen ❄️

LoRA Parameters
        ↓
      Trainable ✓
```

Maybe only a small fraction of parameters are trainable.

---

## QLoRA

```text
7B Parameters
      ↓
Quantized representation
      ↓
Base model frozen ❄️
      +
LoRA adapters
      ↓
Trainable ✓
```

---

# 4. Mathematical difference

## Full Fine-Tuning

Original model:

```text
Y = XW
```

Update:

```text
W_new = W - η × ∇W
```

The complete matrix is updated.

---

## LoRA

Original weight:

```text
W
```

Freeze it:

```text
W = frozen
```

Add a low-rank update:

```text
ΔW = BA
```

The effective weight becomes:

```text
W_new = W + BA
```

Forward pass:

```text
Y = X(W + BA)
```

Train:

```text
A ✓
B ✓
```

Freeze:

```text
W ❄️
```

---

## QLoRA

Conceptually:

```text
W
↓
Quantize
↓
W_4bit
```

Then:

```text
Y = X(W_4bit + BA)
```

Important:

```text
W_4bit → frozen
A → trainable
B → trainable
```

---

# 5. Code comparison

Let's use the same base model.

```python
MODEL_NAME = "gpt2"
```

Install:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes
```

---

# PART 1 — FULL FINE-TUNING

## Step 1: Load model

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

## Step 2: Check trainable parameters

```python
def print_trainable_parameters(model):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(
        f"Trainable %: "
        f"{100 * trainable / total:.4f}%"
    )
```

```python
print_trainable_parameters(model)
```

Output conceptually:

```text
Total: 124,000,000

Trainable: 124,000,000

Trainable %: 100%
```

Because:

```python
for parameter in model.parameters():
    parameter.requires_grad = True
```

---

## Step 3: Optimizer

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)
```

This is full fine-tuning because:

```text
model.parameters()
        ↓
ALL model parameters
        ↓
Optimizer
        ↓
Updated
```

---

# PART 2 — LoRA

Now let's apply LoRA to the same type of model.

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

Load the model:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Create LoRA configuration.

For GPT-2, common target modules include:

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "c_attn"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

Apply LoRA:

```python
lora_model = get_peft_model(
    base_model,
    lora_config
)
```

Check:

```python
lora_model.print_trainable_parameters()
```

Conceptually:

```text
Trainable params:
~300,000

Total params:
~124,000,000

Trainable:
~0.2%
```

Exact numbers depend on model and configuration.

---

## LoRA optimizer

```python
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        lora_model.parameters()
    ),
    lr=2e-4
)
```

Only:

```text
LoRA A → updated ✓

LoRA B → updated ✓
```

Base model:

```text
Attention weights → frozen ❄️
MLP weights       → frozen ❄️
Embeddings        → frozen ❄️
```

---

# PART 3 — QLoRA

QLoRA adds quantization.

```python
import torch

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
```

---

## Step 1: Configure 4-bit quantization

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)
```

---

## Step 2: Load quantized model

```python
qlora_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Now conceptually:

```text
Base Model

FP16 / BF16
      ↓
4-bit quantized representation
```

---

## Step 3: Prepare model for k-bit training

```python
from peft import prepare_model_for_kbit_training

qlora_model = prepare_model_for_kbit_training(
    qlora_model
)
```

---

## Step 4: Add LoRA

```python
from peft import (
    LoraConfig,
    get_peft_model
)

qlora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "c_attn"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

Apply:

```python
qlora_model = get_peft_model(
    qlora_model,
    qlora_config
)
```

Check:

```python
qlora_model.print_trainable_parameters()
```

The result is:

```text
Base model:
4-bit
Frozen ❄️

LoRA adapters:
Trainable ✓
```

---

# 6. Same dataset for all three

Let's create a dataset.

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
        "I was charged twice.",
        "I cannot log in.",
        "My application crashes.",
        "My refund is delayed."
    ],

    "output": [
        "billing",
        "account_access",
        "technical",
        "refund"
    ]
}

dataset = Dataset.from_dict(data)
```

---

# 7. Shared tokenization code

```python
MAX_LENGTH = 256


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


    labels = tokens["input_ids"].copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )


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

This dataset code works for:

```text
Full Fine-Tuning ✓
LoRA ✓
QLoRA ✓
```

The dataset does not fundamentally change.

The difference is:

```text
Which parameters are updated?
```

---

# 8. Training code — Full Fine-Tuning

```python
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

training_args = TrainingArguments(

    output_dir="./full_ft",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=2e-5,

    logging_steps=1,

    report_to="none"
)
```

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
)
```

Train:

```python
trainer.train()
```

Result:

```text
Loss
 ↓
Backpropagation
 ↓

Embedding weights        ✓ Updated
Attention weights       ✓ Updated
MLP weights             ✓ Updated
LayerNorm weights       ✓ Updated
Output layer            ✓ Updated
```

---

# 9. Training code — LoRA

```python
training_args = TrainingArguments(

    output_dir="./lora_ft",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=2e-4,

    logging_steps=1,

    report_to="none"
)
```

```python
trainer = Trainer(

    model=lora_model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=lora_model
    )
)
```

Train:

```python
trainer.train()
```

Internally:

```text
Loss
 ↓
Backpropagation
 ↓

Base Embeddings         ❄️ Frozen
Base Attention          ❄️ Frozen
Base MLP                ❄️ Frozen

LoRA A                  ✓ Updated
LoRA B                  ✓ Updated
```

---

# 10. Training code — QLoRA

```python
training_args = TrainingArguments(

    output_dir="./qlora_ft",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    report_to="none"
)
```

```python
trainer = Trainer(

    model=qlora_model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=qlora_model
    )
)
```

Train:

```python
trainer.train()
```

Internally:

```text
Loss
 ↓
Backpropagation
 ↓

4-bit Base Model        ❄️ Frozen

LoRA A                  ✓ Updated
LoRA B                  ✓ Updated
```

---

# 11. Memory comparison

Imagine a 7B model.

## Full fine-tuning

```text
Model Weights
     +
Gradients
     +
Optimizer States
     +
Activations

= HIGH GPU MEMORY
```

---

## LoRA

```text
Model Weights
     +
Small LoRA Gradients
     +
Small Optimizer States
     +
Activations

= LOWER GPU MEMORY
```

---

## QLoRA

```text
4-bit Model Weights
       +
Small LoRA Gradients
       +
Small Optimizer States
       +
Activations

= LOWEST GPU MEMORY
```

Important:

> Actual GPU memory depends on model size, sequence length, batch size, precision, optimizer, activation checkpointing, and distributed-training strategy.

---

# 12. Save model comparison

## Full fine-tuning

```python
model.save_pretrained(
    "./full_model"
)
```

Conceptually:

```text
./full_model/

config.json
model.safetensors   ← entire model
tokenizer files
```

---

## LoRA

```python
lora_model.save_pretrained(
    "./lora_adapter"
)
```

Conceptually:

```text
./lora_adapter/

adapter_config.json
adapter_model.safetensors
```

Only the adapter weights are stored.

---

## QLoRA

```python
qlora_model.save_pretrained(
    "./qlora_adapter"
)
```

Again, typically:

```text
Base model → loaded separately

QLoRA adapter → saved separately
```

---

# 13. Loading a LoRA adapter

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = PeftModel.from_pretrained(
    base_model,
    "./lora_adapter"
)
```

Conceptually:

```text
Base Model
    +
LoRA Adapter
    ↓
Specialized Model
```

---

# 14. Loading a QLoRA adapter

First load the base model with the same quantization configuration:

```python
base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Then load the adapter:

```python
model = PeftModel.from_pretrained(
    base_model,
    "./qlora_adapter"
)
```

---

# 15. Important hyperparameters

## `r` — LoRA rank

```python
r=8
```

Means the size of the low-rank representation.

```text
Smaller rank:
Less memory
Fewer parameters

Higher rank:
More adaptation capacity
More parameters
```

Typical values:

```text
4
8
16
32
64
```

Start with something like:

```python
r=8
```

or:

```python
r=16
```

then validate experimentally.

---

## `lora_alpha`

```python
lora_alpha=16
```

Controls scaling.

The effective update commonly scales approximately as:

```text
alpha / rank
```

Example:

```text
alpha = 16
rank = 8

Scaling = 2
```

---

## `lora_dropout`

```python
lora_dropout=0.05
```

Adds regularization to help reduce overfitting.

---

# 16. Production decision framework

## Scenario 1: Current knowledge changes daily

Example:

```text
Company policies
Financial prices
Product documentation
```

Don't immediately fine-tune.

Use:

```text
RAG
```

Because:

```text
Documents change
     ↓
Update vector/database index
     ↓
No model retraining
```

---

## Scenario 2: Need a specific response style

Example:

```text
Always return strict JSON
Always use company tone
Follow a workflow
```

Start with:

```text
Prompt Engineering
```

If the behavior must be highly consistent:

```text
LoRA / Fine-tuning
```

---

## Scenario 3: You have a large high-quality dataset and enough GPUs

Example:

```text
Millions of high-quality examples
```

You may consider:

```text
Full Fine-Tuning
```

---

## Scenario 4: You have limited GPU resources

Use:

```text
QLoRA
```

---

## Scenario 5: You need several specialized models

Example:

```text
One base model

Finance Adapter
Legal Adapter
Customer Support Adapter
Coding Adapter
```

Use:

```text
LoRA adapters
```

because:

```text
One Base Model
+
Multiple Small Adapters
```

---

# 17. Real-world architecture

For an enterprise AI system:

```text
                    User
                     │
                     ▼
              API Gateway
                     │
                     ▼
                  FastAPI
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
        RAG System          Model Router
           │                   │
           ▼                   ▼
     Qdrant / Vector DB      Base LLM
                                 │
                 ┌───────────────┼──────────────┐
                 ▼               ▼              ▼
            Finance LoRA     Legal LoRA     Support LoRA
```

Example router:

```python
def select_adapter(user_domain):

    adapters = {
        "finance": "finance_adapter",
        "legal": "legal_adapter",
        "support": "support_adapter"
    }

    return adapters.get(
        user_domain,
        "general_adapter"
    )
```

In a production system, the actual routing may also consider:

```text
Tenant
User permissions
Task type
Model quality
Latency
Cost
```

---

# 18. Interview comparison table

| Feature                   | Full Fine-Tuning          | LoRA            | QLoRA                |
| ------------------------- | ------------------------- | --------------- | -------------------- |
| Base model weights        | Updated                   | Frozen          | Quantized + frozen   |
| Trainable parameters      | All/nearly all            | Small subset    | Small subset         |
| Quantization              | Optional                  | Optional        | Typically 4-bit base |
| GPU memory                | Highest                   | Lower           | Usually lowest       |
| Optimizer states          | For all trainable weights | Mostly adapters | Mostly adapters      |
| Training cost             | Highest                   | Lower           | Lower                |
| Storage                   | Full model                | Small adapters  | Small adapters       |
| Multiple specializations  | Expensive                 | Easy            | Easy                 |
| Setup complexity          | Moderate                  | Moderate        | Higher               |
| Best for limited hardware | No                        | Sometimes       | Yes                  |

---

# 19. The most important interview answer

> **Full fine-tuning updates all or nearly all parameters of the pre-trained model. It provides maximum adaptation capacity but is expensive because gradients and optimizer states are required for the whole model.**

> **LoRA freezes the original model and adds small low-rank trainable matrices to selected layers. This drastically reduces trainable parameters, memory, and storage while preserving the original model.**

> **QLoRA combines LoRA with low-bit quantization of the frozen base model, commonly 4-bit. The base model remains frozen while LoRA adapters are trained, reducing memory further and enabling efficient fine-tuning of larger models on constrained hardware.**

## Remember this diagram

```text
                    FULL FT
                 Update Everything
                        │
                        ▼
               Highest Compute/Memory


                    LoRA
          Freeze Model + Train Adapters
                        │
                        ▼
               Lower Compute/Memory


                    QLoRA
        Quantize + Freeze + Train Adapters
                        │
                        ▼
               Lowest Memory of the three
```

### My practical recommendation

For most real-world LLM projects, evaluate in this order:

```text
Prompt Engineering
       ↓
RAG (if knowledge is the problem)
       ↓
LoRA / QLoRA (if behavior/task specialization is needed)
       ↓
Full Fine-Tuning (only if justified)
```

This is usually the strongest answer in an interview because it shows that you understand **technical trade-offs**, not just definitions.
