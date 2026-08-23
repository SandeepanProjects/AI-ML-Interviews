# What is Parameter-Efficient Fine-Tuning (PEFT)?

**Parameter-Efficient Fine-Tuning (PEFT)** is a technique for adapting a pre-trained LLM to a new task **without updating all of the model's parameters**.

Instead of this:

```text
Base LLM
7B parameters
      ↓
Update all 7B parameters
      ↓
Expensive fine-tuning
```

PEFT does this:

```text
Base LLM
7B parameters
      ↓
Freeze most parameters ❄️
      +
Add/train a small number of parameters
      ↓
Adapted model
```

> **PEFT reduces GPU memory, training time, and storage by training only a small fraction of parameters.**

The most common PEFT technique is **LoRA**.

---

# 1. Why do we need PEFT?

Imagine you have a 7B parameter model.

## Full fine-tuning

```text
7B model
   │
   ├── Model weights
   ├── Gradients
   ├── Optimizer states
   └── Activations
```

During training, GPU memory is required for much more than just the model weights.

Conceptually:

```text
GPU Memory
│
├── Model parameters
├── Gradients
├── Optimizer states
└── Activations
```

For a large model, full fine-tuning can be expensive.

---

## PEFT

With PEFT:

```text
Base Model
    │
    ├── Frozen ❄️
    │
    └── Small trainable modules
             │
             ▼
        Update only these
```

Benefits:

* Lower GPU memory
* Faster training
* Smaller checkpoints
* Multiple task adapters can share one base model

---

# 2. The key idea

Suppose a weight matrix in the base model is:

```text
W
```

During full fine-tuning:

```text
W → W'
```

The entire matrix is updated.

With PEFT such as LoRA:

```text
W is frozen

W' = W + ΔW
```

Instead of directly learning the entire `ΔW`, LoRA approximates it with smaller matrices:

```text
ΔW = B × A
```

where the rank is much smaller than the original dimensions.

For example:

```text
Original weight:

W: 4096 × 4096

Full update:
4096 × 4096
= 16,777,216 parameters


LoRA:

A: 8 × 4096
B: 4096 × 8

Total:
32,768 + 32,768
= 65,536 parameters
```

Instead of training ~16.8 million parameters for that matrix, you train ~65 thousand.

---

# 3. Visual understanding of LoRA

```text
                 Input
                   │
                   ▼
              ┌─────────┐
              │    W    │
              │ Frozen  │
              └─────────┘
                   │
                   ├───────────────┐
                   │               │
                   ▼               ▼
                 W × X          LoRA
                                  │
                                  ▼
                             A × B × X
                                  │
                   ┌──────────────┘
                   ▼
               Add Together
                   │
                   ▼
                 Output
```

Mathematically:

```text
Original:

Y = W × X


LoRA:

Y = W × X + ΔW × X

ΔW = B × A


Therefore:

Y = W × X + B × A × X
```

The original weight `W` stays frozen.

Only:

```text
A and B
```

are trained.

---

# 4. PEFT vs Full Fine-Tuning

| Feature              | Full Fine-Tuning      | PEFT                            |
| -------------------- | --------------------- | ------------------------------- |
| Base parameters      | Updated               | Mostly frozen                   |
| Trainable parameters | Very large            | Small percentage                |
| GPU memory           | High                  | Lower                           |
| Training cost        | High                  | Lower                           |
| Checkpoint size      | Large                 | Small adapters                  |
| Multiple tasks       | Separate model copies | Separate adapters               |
| Typical technique    | Update all weights    | LoRA / adapters / prefix tuning |

---

# 5. Common PEFT techniques

```text
PEFT
│
├── LoRA
│
├── QLoRA
│
├── Adapters
│
├── Prefix Tuning
│
├── Prompt Tuning
│
└── IA³
```

The most important one for interviews today is usually:

> **LoRA**

Then:

> **QLoRA**

Let's understand them with code.

---

# 6. LoRA with Hugging Face PEFT

## Install dependencies

```bash
pip install torch transformers datasets peft trl accelerate
```

---

# 7. Step 1: Load a base model

For demonstration, let's use GPT-2.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# GPT-2 does not define a pad token by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

At this stage:

```text
GPT-2
│
├── Transformer Layer 1
├── Transformer Layer 2
├── Transformer Layer 3
└── ...
```

Normally, full fine-tuning would update many/all trainable parameters.

---

# 8. Step 2: Configure LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Let's understand each parameter.

## `r`

```python
r=8
```

This is the **rank** of the LoRA matrices.

Smaller:

```text
r = 4
```

→ fewer trainable parameters.

Larger:

```text
r = 32
```

→ more adaptation capacity but more memory.

Conceptually:

```text
Original weight
4096 × 4096

LoRA rank = 8

A = 8 × 4096
B = 4096 × 8
```

---

## `lora_alpha`

```python
lora_alpha=16
```

This controls scaling of the LoRA update.

Conceptually:

```text
Effective update:

ΔW × (alpha / rank)
```

Often represented as:

```text
W_new = W + scaling × B × A
```

where:

```text
scaling = alpha / r
```

---

## `lora_dropout`

```python
lora_dropout=0.05
```

Applies dropout to the LoRA path during training.

This can help regularization and reduce overfitting.

---

## `task_type`

```python
task_type="CAUSAL_LM"
```

We are adapting a causal language model.

---

# 9. Step 3: Apply PEFT to the model

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Conceptually, output may look like:

```text
trainable params: 500,000
all params: 124,000,000
trainable%: 0.4%
```

The exact values depend on the model and target modules.

This means:

```text
124 million parameters
       │
       ├── 123.5M frozen
       │
       └── 0.5M trainable
```

That is PEFT.

---

# 10. Where does LoRA get added?

LoRA is usually applied to selected linear layers inside Transformer blocks.

For example, attention projections:

```text
Attention:

Query Projection (Wq)
Key Projection   (Wk)
Value Projection (Wv)
Output Projection(Wo)
```

Conceptually:

```text
Input
 │
 ├── Wq → Query
 ├── Wk → Key
 └── Wv → Value
```

LoRA can add trainable updates to selected projections:

```text
Wq' = Wq + BqAq
Wv' = Wv + BvAv
```

In practice, the exact module names depend on the architecture.

For example, with some models:

```python
LoraConfig(
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

⚠️ **Important:** Do not blindly copy `q_proj` and `v_proj` for every model. The layer names differ between architectures.

You can inspect the model:

```python
for name, module in model.named_modules():
    if "proj" in name:
        print(name)
```

For GPT-2, attention layers use different module names from many LLaMA-family models.

---

# 11. Complete LoRA fine-tuning example

Let's build a small instruction-tuning dataset.

## `data/train.jsonl`

```json
{"instruction":"Classify the customer issue.","input":"I was charged twice.","output":"billing"}
{"instruction":"Classify the customer issue.","input":"I cannot log in.","output":"account_access"}
{"instruction":"Classify the customer issue.","input":"The application crashes.","output":"technical"}
{"instruction":"Classify the customer issue.","input":"My refund has not arrived.","output":"refund"}
```

Now the training code.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model
)


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "gpt2"
MAX_LENGTH = 256


# -----------------------------
# Load dataset
# -----------------------------

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)


# -----------------------------
# Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# Load base model
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# -----------------------------
# LoRA configuration
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# -----------------------------
# Convert model to PEFT model
# -----------------------------

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# -----------------------------
# Tokenize examples
# -----------------------------

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

    # Ignore prompt tokens when calculating loss
    labels[:prompt_length] = [-100] * prompt_length

    full_tokens["labels"] = labels

    return full_tokens


tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)


# -----------------------------
# Data collator
# -----------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# -----------------------------
# Training arguments
# -----------------------------

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)


# -----------------------------
# Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


# -----------------------------
# Train
# -----------------------------

trainer.train()


# -----------------------------
# Save LoRA adapter
# -----------------------------

model.save_pretrained(
    "./lora_adapter"
)

tokenizer.save_pretrained(
    "./lora_adapter"
)
```

---

# 12. What happens internally in this code?

Let's follow one example:

```text
Instruction:
Classify the customer issue.

Input:
I was charged twice.

Output:
billing
```

### Step 1: Tokenization

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

---

### Step 2: Forward pass

```python
outputs = model(
    input_ids=input_ids,
    labels=labels
)
```

Internally:

```text
Input Tokens
      ↓
Base Model
(Frozen Weights)
      +
LoRA Adapters
(Trainable)
      ↓
Predictions
```

---

### Step 3: Loss

The model predicts tokens.

```text
Expected:
billing

Model prediction:
technical
```

Loss measures the error.

```text
Prediction
    ↓
Cross-Entropy Loss
    ↓
Loss = 2.4
```

---

### Step 4: Backpropagation

```python
loss.backward()
```

Gradients are calculated.

But importantly:

```text
Base model weights:
Frozen
No updates

LoRA A/B matrices:
Trainable
Receive gradients
```

Conceptually:

```text
                  Loss
                   │
                   ▼
            Backpropagation
                   │
          ┌────────┴────────┐
          │                 │
     Base Weights        LoRA Weights
       Frozen ❄️          Trainable ✓
          │                 │
          ✗                 ▼
                         Update
```

---

### Step 5: Optimizer update

Conceptually:

```python
optimizer.step()
```

Only LoRA parameters change.

```text
Before:

W_base = frozen
A = random values
B = random values

After:

W_base = unchanged
A = updated
B = updated
```

After training:

```text
Base Model
    +
Trained LoRA Adapter
    ↓
Task-specific model behavior
```

---

# 13. How do we load the trained LoRA adapter?

Important: usually, the adapter alone is **not a complete standalone model**.

You load:

```text
Original Base Model
       +
LoRA Adapter
       ↓
Fine-tuned Model
```

Code:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel


MODEL_NAME = "gpt2"
ADAPTER_PATH = "./lora_adapter"


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
```

Now generate:

```python
import torch


prompt = """### Instruction:
Classify the customer issue.

### Input:
I was charged twice on my credit card.

### Response:
"""


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )


response = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print(response)
```

Conceptually:

```text
### Response:
billing
```

---

# 14. Merging LoRA with the base model

Sometimes you want to merge the adapter into the base model.

```python
merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    "./merged_model"
)

tokenizer.save_pretrained(
    "./merged_model"
)
```

Conceptually:

Before:

```text
Base Model
     +
LoRA Adapter
```

After merge:

```text
Merged Model

W_merged = W_base + ΔW
```

Then inference can load one merged model.

Whether you should merge depends on your serving/deployment setup.

---

# 15. What is QLoRA?

Now let's go one step further.

## LoRA

```text
Base Model
   ↓
Normal precision
   ↓
Frozen
   +
LoRA adapters
   ↓
Train adapters
```

## QLoRA

```text
Base Model
   ↓
Quantize to lower-bit representation
   ↓
Frozen
   +
LoRA adapters
   ↓
Train adapters
```

The purpose:

> **Reduce memory requirements further.**

---

# 16. QLoRA code

Install:

```bash
pip install bitsandbytes
```

Load a quantized model:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


MODEL_NAME = "your-base-model"


quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
```

Then prepare it for PEFT training:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

Add LoRA:

```python
from peft import (
    LoraConfig,
    get_peft_model
)


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


model = get_peft_model(
    model,
    lora_config
)
```

Now:

```text
4-bit Quantized Base Model
             +
        LoRA Adapters
             ↓
       Train Adapters
```

This is QLoRA.

---

# 17. Important: LoRA vs QLoRA

| Feature             | LoRA                       | QLoRA                     |
| ------------------- | -------------------------- | ------------------------- |
| Base model          | Usually standard precision | Quantized, commonly 4-bit |
| LoRA adapters       | Yes                        | Yes                       |
| Base model updated  | No                         | No                        |
| Memory usage        | Lower than full FT         | Even lower                |
| Training complexity | Simpler                    | More complex              |
| Best use            | Enough GPU memory          | Limited GPU memory        |

---

# 18. Other PEFT methods

## A. Adapter Tuning

Insert small neural network layers into the Transformer.

```text
Transformer Layer
       ↓
   Adapter Layer
       ↓
Transformer Layer
```

Only adapters are trained.

---

## B. Prompt Tuning

Instead of changing model weights, learn virtual embeddings:

```text
Learned Soft Prompt
        +
User Input
        ↓
Frozen LLM
```

Conceptually:

```text
[Learned Tokens] + "Summarize this..."
```

---

## C. Prefix Tuning

Adds trainable prefix representations to Transformer layers.

```text
Trainable Prefix
       ↓
Attention Layers
       +
Input
```

---

## D. IA³

Uses learned scaling vectors to modify activations inside selected model components.

---

# 19. When should you use PEFT?

Use PEFT when:

### You have limited GPU resources

```text
Large Model
+
Limited GPU
=
PEFT / LoRA / QLoRA
```

### You need multiple specialized models

For example:

```text
                    Base LLM
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
 Finance Adapter   Legal Adapter   Support Adapter
```

Instead of storing three full copies of the model, you can store:

```text
1 Base Model
+
3 Small Adapters
```

### You want fast experimentation

```text
Dataset Version 1
       ↓
LoRA Adapter V1

Dataset Version 2
       ↓
LoRA Adapter V2
```

Much easier than storing multiple full models.

---

# 20. Production architecture example

For an enterprise system:

```text
                     API
                      │
                      ▼
                Model Router
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Finance      Legal       Support
       Adapter      Adapter      Adapter
          │           │           │
          └───────────┼───────────┘
                      ▼
                   Base LLM
```

Conceptually, one base model can support multiple specialized behaviors through different adapters, depending on serving framework and adapter-loading capabilities.

---

# 21. Common interview questions

### Why is PEFT more memory efficient?

Because:

```text
Full Fine-tuning:

Base Weights
+ Gradients
+ Optimizer States
for many/all parameters


PEFT:

Frozen Base Weights
+
Gradients/Optimizer States
only for small trainable parameters
```

---

### Does PEFT modify the base model?

Usually:

```text
Base Model → Frozen
PEFT Modules → Updated
```

So the base model remains unchanged while adapters store task-specific changes.

---

### Is LoRA the same as fine-tuning?

LoRA is a **method of fine-tuning**.

```text
Fine-tuning
│
├── Full fine-tuning
│
└── PEFT
     │
     ├── LoRA
     ├── QLoRA
     ├── Adapters
     └── Prompt tuning
```

---

# Best interview answer

> **"Parameter-efficient fine-tuning, or PEFT, is a family of techniques used to adapt a pre-trained model while updating only a small subset of parameters instead of the entire model. This reduces GPU memory, training cost, optimizer-state memory, and checkpoint size. A common method is LoRA, where the original model weights remain frozen and the weight update is represented by low-rank trainable matrices. In practice, I would load a base model, configure LoRA using the PEFT library, train it on supervised examples, evaluate it on held-out data, and save the adapter separately. If GPU memory is constrained, I would use QLoRA, which combines a quantized base model with LoRA adapters."**

## The one diagram to remember

```text
FULL FINE-TUNING

Base Model
     │
     ▼
Update ALL weights
     │
     ▼
Large GPU memory


PEFT / LoRA

Base Model
     │
     ├── Frozen ❄️
     │
     └── LoRA adapters
             │
             ▼
         Update only
             │
             ▼
       Lower cost
```

> **PEFT = Keep the large model mostly unchanged and train only a small number of additional parameters.**
