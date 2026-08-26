# What is PEFT?

**PEFT = Parameter-Efficient Fine-Tuning.**

Instead of updating all parameters of a large pretrained model, PEFT:

1. Keeps most of the base model frozen.
2. Adds a small number of trainable parameters.
3. Trains only those parameters.

The main Hugging Face implementation is [PEFT](https://huggingface.co/docs/peft/index?utm_source=chatgpt.com).

---

# 1. Why do we need PEFT?

Suppose we have an LLM with:

```text
Llama Model
= 8 billion parameters
```

## Full fine-tuning

```text
8B parameters
       ↓
All parameters trainable
       ↓
Huge GPU memory
       ↓
Large optimizer states
       ↓
Expensive training
```

Conceptually:

```python
for parameter in model.parameters():
    parameter.requires_grad = True
```

Every parameter gets gradients:

```text
Base Model Weights
████████████████████████████
████████████████████████████
████████████████████████████

All updated
```

---

## PEFT

With PEFT:

```text
Base Model
████████████████████████████
████████████████████████████
████████████████████████████

Mostly FROZEN ❄️

          +

Small trainable components
██
██
██
```

For example:

```text
Total parameters:      8,000,000,000

Trainable parameters:     20,000,000

Percentage:                 0.25%
```

The exact numbers depend on the model and PEFT configuration.

---

# 2. What does PEFT actually mean?

PEFT is a **family of fine-tuning techniques**.

```text
                    PEFT
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
     LoRA         Adapters      Prompt Tuning
       │
       ▼
     QLoRA
```

Common PEFT approaches include:

| Method         | Main idea                                    |
| -------------- | -------------------------------------------- |
| LoRA           | Train low-rank matrices                      |
| QLoRA          | Quantized base model + LoRA                  |
| Adapter tuning | Add small neural modules                     |
| Prompt tuning  | Train virtual prompt embeddings              |
| Prefix tuning  | Train prefix vectors                         |
| IA³            | Scale activations with small learned vectors |

The [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index?utm_source=chatgpt.com) provides implementations for multiple parameter-efficient adaptation methods.

---

# 3. How does PEFT work?

Imagine the base model:

```text
                    LLM
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      Layer 1      Layer 2      Layer N
         │           │           │
         ▼           ▼           ▼
       Frozen      Frozen       Frozen
```

PEFT adds small trainable components:

```text
                  LLM
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
    Layer 1     Layer 2       Layer N
       │           │            │
       ▼           ▼            ▼
     Frozen      Frozen       Frozen
       │           │
       ▼           ▼
    Adapter      Adapter
    Trainable    Trainable
```

During training:

```text
Input
  │
  ▼
Frozen Base Model
  │
  ▼
Trainable PEFT Layers
  │
  ▼
Loss
  │
  ▼
Backpropagation
  │
  ▼
Update ONLY PEFT parameters
```

---

# 4. What is LoRA in PEFT?

LoRA is one of the most popular PEFT techniques.

Normally, a neural network has a weight matrix:

$$
W
$$

During full fine-tuning:

$$
W_{new} = W + \Delta W
$$

where the full matrix \(\Delta W\) must be learned.

LoRA approximates the update using two small matrices:

$$
\Delta W = B A
$$

So:

$$
W_{new} = W + \frac{\alpha}{r}BA
$$

Where:

* `W` = frozen pretrained weight
* `A` = trainable low-rank matrix
* `B` = trainable low-rank matrix
* `r` = LoRA rank
* `alpha` = scaling factor

---

# 5. Why does LoRA reduce parameters?

Suppose:

```text
Original weight:

W = 4096 × 4096
```

Full matrix parameters:

$$
4096 \times 4096 = 16,777,216
$$

So full fine-tuning updates:

```text
16.7 million parameters
```

Now use LoRA rank:

```text
r = 8
```

LoRA matrices:

$$
A = 8 \times 4096
$$

$$
B = 4096 \times 8
$$

Total:

$$
8 \times 4096 + 4096 \times 8
$$

$$
65,536
$$

Compare:

```text
Full fine-tuning:
16,777,216 parameters

LoRA:
65,536 parameters
```

That is dramatically fewer trainable parameters.

---

# 6. Install PEFT

```bash
pip install peft transformers accelerate
```

For QLoRA:

```bash
pip install bitsandbytes
```

---

# 7. Basic LoRA configuration using PEFT

The core PEFT object is:

```python
from peft import LoraConfig
```

Example:

```python
from peft import LoraConfig


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

Let's understand every parameter.

---

# 8. `r` — LoRA rank

```python
r=16
```

This controls the size of the low-rank matrices.

Original:

```text
W = 4096 × 4096
```

LoRA:

```text
A = 16 × 4096

B = 4096 × 16
```

### Small rank

```python
r=4
```

```text
Pros:
✓ Very few parameters
✓ Low memory

Cons:
✗ Lower adaptation capacity
```

### Medium rank

```python
r=16
```

```text
Good balance
```

### Large rank

```python
r=64
```

```text
Pros:
✓ More capacity

Cons:
✗ More parameters
✗ More memory
✗ More compute
✗ Higher overfitting risk
```

Typical experimentation:

```python
r = 8
r = 16
r = 32
```

---

# 9. `lora_alpha`

```python
lora_alpha=32
```

The LoRA update is scaled:

$$
W_{new}
=
W
+
\frac{\alpha}{r}BA
$$

For:

```python
r = 16
alpha = 32
```

Scaling:

```text
alpha / r

32 / 16

= 2
```

Python:

```python
r = 16
alpha = 32

scaling = alpha / r

print(scaling)
```

Output:

```text
2.0
```

Conceptually:

```text
BA
 │
 ▼
× Scaling
 │
 ▼
LoRA Update
```

---

# 10. `lora_dropout`

```python
lora_dropout=0.05
```

This applies dropout to the LoRA training path.

Conceptually:

```text
Input
  │
  ├──────────────► Frozen W
  │
  ▼
LoRA Dropout
  │
  ▼
A
  │
  ▼
B
  │
  ▼
LoRA Update
```

It can help regularization.

Typical values:

```python
lora_dropout = 0.0
lora_dropout = 0.05
lora_dropout = 0.1
```

For a large, diverse dataset:

```text
0.0 or 0.05
```

For a small dataset where overfitting is a concern:

```text
0.05 or 0.1
```

---

# 11. `target_modules`

This is one of the most important settings.

Example:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

These usually correspond to attention projection layers.

Attention architecture:

```text
Input
  │
  ├────► Q Projection
  │
  ├────► K Projection
  │
  └────► V Projection
           │
           ▼
       Attention
           │
           ▼
       O Projection
```

So:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

means:

> Add LoRA adapters to these modules.

---

# 12. Why target attention layers?

A Transformer layer contains:

```text
Transformer Block
│
├── Attention
│   ├── q_proj
│   ├── k_proj
│   ├── v_proj
│   └── o_proj
│
└── MLP
    ├── gate_proj
    ├── up_proj
    └── down_proj
```

Common LoRA configurations:

### Minimal

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

### Standard

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

### More adaptation capacity

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

The exact names depend on the model architecture, so you should inspect the actual model modules rather than blindly copying names.

---

# 13. How do you apply LoRA using PEFT?

First load a model:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Create the LoRA configuration:

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

    bias="none",

    task_type="CAUSAL_LM"
)
```

Now apply PEFT:

```python
from peft import get_peft_model


model = get_peft_model(
    model,
    lora_config
)
```

This transforms:

```text
Before:

Llama
  │
  ├── q_proj
  ├── k_proj
  ├── v_proj
  └── o_proj
```

Into conceptually:

```text
Llama
  │
  ├── q_proj
  │     └── LoRA A + B
  │
  ├── k_proj
  │     └── LoRA A + B
  │
  ├── v_proj
  │     └── LoRA A + B
  │
  └── o_proj
        └── LoRA A + B
```

The base weights remain frozen.

---

# 14. Check trainable parameters

PEFT provides a useful method:

```python
model.print_trainable_parameters()
```

Example conceptual output:

```text
trainable params: 8,388,608

all params: 3,218,000,000

trainable%: 0.26%
```

This is a very useful thing to show in an interview.

You can also manually calculate it:

```python
def count_parameters(model):

    trainable = 0
    total = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    percentage = (
        trainable / total
    ) * 100

    print(
        f"Trainable: {trainable:,}"
    )

    print(
        f"Total: {total:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )
```

Run:

```python
count_parameters(model)
```

---

# 15. Complete LoRA training example with PEFT

Let's combine PEFT + Transformers `Trainer`.

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)
```

---

## Step 1: Configuration

```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

OUTPUT_DIR = "./llama-lora"
```

---

## Step 2: Create dataset

```python
dataset = Dataset.from_dict({

    "text": [

        (
            "### Instruction:\n"
            "Explain RAG.\n\n"
            "### Response:\n"
            "RAG retrieves relevant documents "
            "and provides them to an LLM as context."
        ),

        (
            "### Instruction:\n"
            "Explain LoRA.\n\n"
            "### Response:\n"
            "LoRA is a parameter-efficient fine-tuning "
            "method that trains low-rank matrices."
        )
    ]
})
```

---

## Step 3: Load tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

---

## Step 4: Tokenize

```python
def tokenize_function(example):

    return tokenizer(

        example["text"],

        truncation=True,

        max_length=512
    )
```

Apply:

```python
tokenized_dataset = dataset.map(

    tokenize_function,

    batched=True,

    remove_columns=["text"]
)
```

---

## Step 5: Load base model

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16
)
```

---

## Step 6: Configure LoRA

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

---

## Step 7: Convert base model to PEFT model

```python
model = get_peft_model(

    model,

    lora_config
)
```

Check:

```python
model.print_trainable_parameters()
```

Conceptually:

```text
Base Model Parameters
██████████████████████████████
             FROZEN


LoRA Parameters
██
██
TRAINABLE
```

---

## Step 8: Data collator

For causal language modeling:

```python
data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=False
)
```

Why?

```text
Masked Language Modeling:

The cat is [MASK]

BERT
```

vs:

```text
Causal Language Modeling:

The cat is
        ↓
      sleeping

Llama
```

Llama uses:

```python
mlm=False
```

---

## Step 9: Training arguments

```python
training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=10,

    save_strategy="epoch",

    bf16=True,

    report_to="none"
)
```

Notice:

```python
learning_rate=2e-4
```

This is often higher than full fine-tuning because only a small number of LoRA parameters are being trained. You should still tune it based on your dataset and validation results.

---

## Step 10: Create `Trainer`

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)
```

---

## Step 11: Train

```python
trainer.train()
```

During training:

```text
Input
  │
  ▼
Frozen Llama
  │
  ├──── Frozen weights ❄️
  │
  ▼
LoRA A and B
  │
  ├──── Trainable ✓
  │
  ▼
Loss
  │
  ▼
Backpropagation
  │
  ▼
Update A and B only
```

---

# 16. Better approach: PEFT + `SFTTrainer`

For instruction/chat fine-tuning, I would generally prefer `SFTTrainer`.

```python
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig
)
```

Dataset:

```python
dataset = Dataset.from_list([
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Explain PEFT."
            },
            {
                "role": "assistant",
                "content": (
                    "PEFT is Parameter-Efficient Fine-Tuning. "
                    "It fine-tunes a small number of parameters "
                    "while keeping most of the pretrained model frozen."
                )
            }
        ]
    }
])
```

Load:

```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

LoRA:

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

Training configuration:

```python
training_args = SFTConfig(

    output_dir="./llama-peft",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    max_length=1024,

    bf16=True,

    logging_steps=10,

    report_to="none"
)
```

Create trainer:

```python
trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    args=training_args,

    processing_class=tokenizer,

    peft_config=peft_config
)
```

Train:

```python
trainer.train()
```

This is the clean architecture:

```text
                   Llama
                     │
                     ▼
                 PEFT/LoRA
                     │
                     ▼
                 SFTTrainer
                     │
                     ▼
              Fine-tuned Adapter
```

---

# 17. How does PEFT know which parameters to freeze?

Conceptually, before PEFT:

```python
for parameter in model.parameters():
    parameter.requires_grad = True
```

After applying LoRA, conceptually:

```python
for name, parameter in model.named_parameters():

    if "lora" in name:
        parameter.requires_grad = True

    else:
        parameter.requires_grad = False
```

PEFT handles this automatically.

You should **not manually freeze/unfreeze parameters unless you have a specific advanced requirement**.

Check:

```python
for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(name)
```

You will see LoRA-related trainable parameters.

---

# 18. Saving the PEFT adapter

After training:

```python
trainer.save_model(
    "./llama-support-adapter"
)
```

Typically:

```text
llama-support-adapter/
│
├── adapter_config.json
├── adapter_model.safetensors
└── other training/tokenizer files
```

The key idea:

```text
Base Model
8 GB / 16 GB / more
      +
Adapter
Small additional file
```

This makes it easy to maintain multiple task-specific adapters:

```text
                Llama Base Model
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    Support LoRA    SQL LoRA     Coding LoRA
```

---

# 19. Loading the trained adapter

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
```

Load base model:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Load adapter:

```python
model = PeftModel.from_pretrained(

    base_model,

    "./llama-support-adapter"
)
```

Now:

```text
Base Model
     +
LoRA Adapter
     ↓
Fine-tuned Model
```

---

# 20. Complete production-style PEFT configuration

A practical starting configuration for Llama SFT:

```python
from peft import LoraConfig


peft_config = LoraConfig(

    # Adapter capacity
    r=16,

    # Scaling
    lora_alpha=32,

    # Regularization
    lora_dropout=0.05,

    # Attention projections
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    # Do not train biases
    bias="none",

    # Causal language modeling
    task_type="CAUSAL_LM"
)
```

For broader adaptation:

```python
peft_config = LoraConfig(

    r=32,

    lora_alpha=64,

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

# 21. How do I choose the LoRA configuration?

A practical starting strategy:

### Small/simple dataset

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
```

### Normal instruction tuning

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
    ]
)
```

### Complex domain adaptation

```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

Then compare validation results.

Do **not** assume:

```text
Higher rank = always better
```

Higher rank means:

```text
More trainable parameters
        +
More capacity
        +
More memory
        +
Potential overfitting
```

---

# 22. PEFT vs Full Fine-Tuning

| Feature                 | Full Fine-Tuning     | PEFT / LoRA              |
| ----------------------- | -------------------- | ------------------------ |
| Base weights            | Updated              | Mostly frozen            |
| Trainable parameters    | 100%                 | Small percentage         |
| GPU memory              | Very high            | Lower                    |
| Training cost           | High                 | Lower                    |
| Training speed          | Depends              | Often more practical     |
| Adapter storage         | Full model           | Small adapter            |
| Multiple tasks          | Multiple full models | One base + many adapters |
| Catastrophic forgetting | Higher risk          | Often reduced            |

---

# 23. PEFT vs LoRA

This is important in interviews:

```text
PEFT
 │
 ├── LoRA
 ├── QLoRA
 ├── Adapter Tuning
 ├── Prefix Tuning
 ├── Prompt Tuning
 └── Other efficient adaptation methods
```

Therefore:

> **PEFT is the broad approach/category (and also the Hugging Face library), while LoRA is one specific parameter-efficient fine-tuning technique implemented by PEFT.**

---

# 24. Interview-ready answer

> **PEFT stands for Parameter-Efficient Fine-Tuning. Instead of updating all parameters of a large pretrained model, PEFT freezes most of the base model and trains a small set of additional parameters. This significantly reduces GPU memory, optimizer memory, training cost, and storage requirements.**
>
> **A common PEFT technique is LoRA. In Hugging Face PEFT, I configure LoRA using `LoraConfig`, where I define the rank `r`, scaling factor `lora_alpha`, dropout, and the target modules such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`. I then either apply it using `get_peft_model()` or pass the configuration directly to TRL's `SFTTrainer`. During training, the base Llama weights remain mostly frozen while the LoRA adapter weights are updated.**

### Most important code to remember

```python
from peft import LoraConfig, get_peft_model


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

Then train the PEFT model with `Trainer` or, for LLM instruction/conversation fine-tuning, preferably `SFTTrainer`.
