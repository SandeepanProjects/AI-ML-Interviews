# PEFT vs Full Fine-Tuning

The fundamental difference is **which parameters you train**.

> **Full fine-tuning updates the original model weights.**
> **PEFT (Parameter-Efficient Fine-Tuning) freezes most original weights and trains only a small number of parameters.**

---

# 1. High-level comparison

```text
FULL FINE-TUNING

Pretrained LLM
     │
     ▼
All model parameters
Trainable ✓
     │
     ▼
Backpropagation
     │
     ▼
All weights updated
```

```text
PEFT

Pretrained LLM
     │
     ▼
Base model frozen ❄️
     │
     ├── LoRA matrices       ✓
     ├── Adapters            ✓
     ├── Soft prompts        ✓
     └── Prefix parameters   ✓
```

---

# 2. What is PEFT?

**PEFT is not one technique.**

It is a family of techniques that reduce the number of trainable parameters.

```text
PEFT
│
├── LoRA
├── QLoRA
├── Adapter Tuning
├── Prompt Tuning
├── Prefix Tuning
└── P-Tuning
```

For example, with a 7B model:

```text
Full Fine-Tuning
----------------
7,000,000,000 parameters
All trainable

PEFT
----
7,000,000,000 base parameters → mostly frozen

+
Small additional trainable parameters
```

---

# 3. Parameter-level difference

Suppose a model has:

```text
Model
│
├── Embedding
├── Transformer Layer 1
│    ├── Attention
│    └── MLP
├── Transformer Layer 2
│    ├── Attention
│    └── MLP
└── Output Layer
```

## Full fine-tuning

```text
Embedding               ✓ Train
Attention               ✓ Train
MLP                     ✓ Train
LayerNorm               ✓ Train
Output Layer            ✓ Train
```

## PEFT

For example, LoRA:

```text
Embedding               ❄️ Frozen
Attention base weights  ❄️ Frozen
MLP base weights        ❄️ Frozen
LayerNorm               ❄️ Usually frozen

LoRA adapters           ✓ Train
```

---

# 4. Full fine-tuning from scratch in PyTorch

Let's first build a simple model.

```python
import torch
import torch.nn as nn
```

```python
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(10, 64)

        self.layer2 = nn.Linear(64, 64)

        self.output = nn.Linear(64, 3)

    def forward(self, x):

        x = torch.relu(
            self.layer1(x)
        )

        x = torch.relu(
            self.layer2(x)
        )

        return self.output(x)
```

Create the model:

```python
model = SimpleModel()
```

---

## Check parameters

```python
def print_parameters(model):

    total = 0
    trainable = 0

    for name, param in model.named_parameters():

        total += param.numel()

        if param.requires_grad:
            trainable += param.numel()

        print(
            f"{name:25} "
            f"trainable={param.requires_grad}"
        )

    print("-" * 40)

    print("Total:", total)

    print("Trainable:", trainable)
```

Run:

```python
print_parameters(model)
```

Output conceptually:

```text
layer1.weight             trainable=True
layer1.bias               trainable=True
layer2.weight             trainable=True
layer2.bias               trainable=True
output.weight             trainable=True
output.bias               trainable=True

Total: XXXXX
Trainable: XXXXX
```

Everything is trainable.

---

# 5. Full fine-tuning training

```python
from torch.optim import AdamW
```

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3
)
```

The important line is:

```python
model.parameters()
```

This passes **all parameters** to the optimizer.

Training:

```python
loss_function = nn.CrossEntropyLoss()

model.train()

for epoch in range(10):

    # Example training data
    x = torch.randn(
        32,
        10
    )

    labels = torch.randint(
        0,
        3,
        (32,)
    )

    # Forward pass
    predictions = model(x)

    # Calculate loss
    loss = loss_function(
        predictions,
        labels
    )

    # Remove previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update ALL trainable parameters
    optimizer.step()

    print(
        f"Epoch {epoch}: "
        f"{loss.item():.4f}"
    )
```

Internally:

```text
loss.backward()
       │
       ▼
Gradients calculated for:

layer1 ✓
layer2 ✓
output ✓
       │
       ▼
optimizer.step()
       │
       ▼
All weights updated ✓
```

---

# 6. Now implement PEFT

Let's simulate PEFT using an adapter.

Our architecture:

```text
INPUT
  │
  ▼
BASE MODEL ❄️
  │
  ▼
ADAPTER ✓
  │
  ▼
OUTPUT
```

---

## Step 1: Create a base model

```python
class BaseModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(
            10,
            64
        )

        self.layer2 = nn.Linear(
            64,
            64
        )

    def forward(self, x):

        x = torch.relu(
            self.layer1(x)
        )

        x = torch.relu(
            self.layer2(x)
        )

        return x
```

---

## Step 2: Create an adapter

```python
class Adapter(nn.Module):

    def __init__(self):

        super().__init__()

        # Bottleneck
        self.down = nn.Linear(
            64,
            8
        )

        self.up = nn.Linear(
            8,
            64
        )

    def forward(self, x):

        residual = x

        x = torch.relu(
            self.down(x)
        )

        x = self.up(x)

        return residual + x
```

The adapter performs:

```text
64
 ↓
8
 ↓
64
```

---

## Step 3: Combine them

```python
class PEFTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.base_model = BaseModel()

        self.adapter = Adapter()

        self.output = nn.Linear(
            64,
            3
        )

    def forward(self, x):

        # Base model
        x = self.base_model(x)

        # Adapter
        x = self.adapter(x)

        # Output
        x = self.output(x)

        return x
```

Create it:

```python
peft_model = PEFTModel()
```

---

# 7. Freeze the base model

This is the most important PEFT concept.

```python
for parameter in peft_model.base_model.parameters():

    parameter.requires_grad = False
```

Now check:

```python
print_parameters(peft_model)
```

Conceptually:

```text
base_model.layer1.weight    False ❄️
base_model.layer1.bias      False ❄️

base_model.layer2.weight    False ❄️
base_model.layer2.bias      False ❄️

adapter.down.weight         True ✓
adapter.down.bias           True ✓

adapter.up.weight           True ✓
adapter.up.bias             True ✓

output.weight               True ✓
output.bias                 True ✓
```

---

# 8. Train only PEFT parameters

Instead of:

```python
optimizer = AdamW(
    peft_model.parameters()
)
```

we explicitly select trainable parameters:

```python
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        peft_model.parameters()
    ),
    lr=1e-3
)
```

Now train:

```python
loss_function = nn.CrossEntropyLoss()

peft_model.train()

for epoch in range(10):

    x = torch.randn(
        32,
        10
    )

    labels = torch.randint(
        0,
        3,
        (32,)
    )

    predictions = peft_model(x)

    loss = loss_function(
        predictions,
        labels
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch}: "
        f"{loss.item():.4f}"
    )
```

Now:

```text
loss.backward()
      │
      ▼
Base model ❄️
No trainable gradients for its weights

Adapter ✓
Gradients calculated

Output head ✓
Gradients calculated
      │
      ▼
optimizer.step()
      │
      ▼
Only PEFT parameters updated
```

---

# 9. Verify that base weights don't change

This is a great interview-level demonstration.

```python
import copy

model = PEFTModel()

# Freeze base
for param in model.base_model.parameters():
    param.requires_grad = False


# Save original base weights
original_weight = (
    model.base_model.layer1.weight
    .detach()
    .clone()
)
```

Train the model:

```python
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-3
)

x = torch.randn(32, 10)

labels = torch.randint(
    0,
    3,
    (32,)
)

predictions = model(x)

loss = nn.CrossEntropyLoss()(
    predictions,
    labels
)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Check:

```python
base_changed = torch.equal(
    original_weight,
    model.base_model.layer1.weight
)

print(
    "Base weights unchanged:",
    base_changed
)
```

Expected:

```text
Base weights unchanged: True
```

Now compare an adapter:

```python
adapter_before = (
    model.adapter.down.weight
    .detach()
    .clone()
)

# Run another training step
```

After training:

```python
adapter_changed = not torch.equal(
    adapter_before,
    model.adapter.down.weight
)

print(
    "Adapter weights changed:",
    adapter_changed
)
```

Expected:

```text
Adapter weights changed: True
```

This is PEFT in practice.

---

# 10. PEFT using LoRA with a real Hugging Face model

Now let's use an actual LLM.

Install:

```bash
pip install torch transformers peft accelerate datasets
```

---

## Load the model

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

---

## Check full fine-tuning parameters

```python
total = sum(
    p.numel()
    for p in model.parameters()
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total: {total:,}"
)

print(
    f"Trainable: {trainable:,}"
)
```

Conceptually:

```text
Total: 124,439,808

Trainable: 124,439,808
```

```text
100% trainable
```

This is full fine-tuning.

---

# 11. Convert it to LoRA PEFT

```python
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
```

Configuration:

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "c_attn"
    ],

    bias="none",

    task_type=TaskType.CAUSAL_LM
)
```

Apply:

```python
lora_model = get_peft_model(
    model,
    lora_config
)
```

Check:

```python
lora_model.print_trainable_parameters()
```

Conceptually:

```text
trainable params:
300,000

all params:
124,000,000

trainable %:
~0.2%
```

Exact values depend on the model and PEFT configuration.

Now:

```text
Base GPT-2         ❄️ Frozen

LoRA matrices A    ✓ Trainable
LoRA matrices B    ✓ Trainable
```

---

# 12. Full fine-tuning vs LoRA code

## Full Fine-Tuning

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)
```

Result:

```text
All model weights → Updated
```

---

## PEFT with LoRA

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

lora_model = get_peft_model(
    model,
    lora_config
)

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        lora_model.parameters()
    ),
    lr=2e-4
)
```

Result:

```text
Base weights → Frozen

LoRA A/B → Updated
```

---

# 13. Memory difference

## Full Fine-Tuning

Training memory includes:

```text
Model weights
     +
Gradients for model
     +
Optimizer states
     +
Activations
```

```text
7B model

Model weights         ██████████
Gradients             ██████████
Optimizer states      ███████████████
Activations           ███████

TOTAL                 VERY HIGH
```

---

## PEFT

```text
Frozen Base Model
       +
Small trainable parameters
       +
Small optimizer state
       +
Activations
```

```text
7B model

Model weights         ██████████
PEFT gradients        █
PEFT optimizer        █
Activations           ███████

TOTAL                 MUCH LOWER
```

Note: exact memory usage depends on precision, optimizer, sequence length, batch size, activation checkpointing, and distributed training.

---

# 14. Storage difference

## Full fine-tuning

If your base model is:

```text
7 GB
```

And you fine-tune it for:

```text
Finance
Legal
Support
```

You may store:

```text
Finance Model  → 7 GB+
Legal Model    → 7 GB+
Support Model  → 7 GB+
```

---

## PEFT

```text
Base Model
   ↓
7 GB

Finance LoRA Adapter
   ↓
Small

Legal LoRA Adapter
   ↓
Small

Support LoRA Adapter
   ↓
Small
```

```text
Base Model
+
Multiple small adapters
```

This is especially useful for multi-tenant systems.

---

# 15. Saving: Full Fine-Tuning vs PEFT

## Full Fine-Tuning

```python
model.save_pretrained(
    "./full_finetuned_model"
)

tokenizer.save_pretrained(
    "./full_finetuned_model"
)
```

You save the entire fine-tuned model.

---

## PEFT / LoRA

```python
lora_model.save_pretrained(
    "./lora_adapter"
)
```

Typically you save the adapter weights and configuration.

Later:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

model = PeftModel.from_pretrained(
    base_model,
    "./lora_adapter"
)
```

```text
Base Model
     +
Adapter
     ↓
Fine-tuned behavior
```

---

# 16. Complete comparison

| Feature                 | Full Fine-Tuning              | PEFT                                        |
| ----------------------- | ----------------------------- | ------------------------------------------- |
| Meaning                 | Update full model             | Train a small subset/additional parameters  |
| Base weights            | Updated                       | Mostly frozen                               |
| Trainable parameters    | 100% or nearly 100%           | Small percentage                            |
| GPU memory              | Very high                     | Lower                                       |
| Optimizer states        | Large                         | Small                                       |
| Training cost           | High                          | Lower                                       |
| Training time           | Usually higher                | Usually lower                               |
| Storage                 | Full model per specialization | Small adapter per specialization            |
| Multiple tasks          | Expensive                     | Efficient                                   |
| Catastrophic forgetting | Higher risk                   | Often lower risk                            |
| Example methods         | Full FT                       | LoRA, QLoRA, Adapters, Prompt/Prefix Tuning |

---

# 17. When should you use Full Fine-Tuning?

Use it when:

### You need deep adaptation

```text
Base model
   ↓
Very specialized domain/task
   ↓
Large, high-quality dataset
```

### You have significant compute

```text
Multiple GPUs
Large GPU memory
Distributed training
```

### PEFT does not achieve your required quality

A reasonable evaluation approach:

```text
Prompt engineering
       ↓
RAG (if knowledge is the problem)
       ↓
PEFT
       ↓
Full Fine-Tuning
```

---

# 18. When should you use PEFT?

Use PEFT when:

```text
✓ GPU resources are limited
✓ You have a large model
✓ You need fast experimentation
✓ You need multiple task/domain models
✓ You want low storage cost
✓ You want easy rollback
✓ You want multi-tenant specialization
```

For many practical LLM applications, **PEFT is the first fine-tuning approach to evaluate**.

---

# 19. Real-world example

Imagine an enterprise AI platform:

```text
                      User
                       │
                       ▼
                    FastAPI
                       │
                       ▼
                   Task Router
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Finance       Legal       Support
       Adapter       Adapter     Adapter
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Shared Base LLM
```

A simple adapter selection example:

```python
ADAPTERS = {
    "finance": "finance_adapter",
    "legal": "legal_adapter",
    "support": "support_adapter"
}


def select_adapter(domain: str) -> str:

    return ADAPTERS.get(
        domain,
        "general_adapter"
    )
```

The system can:

1. Load one base model.
2. Select the appropriate PEFT adapter.
3. Generate a response.
4. Switch adapters for different tasks or tenants.

---

# 20. Interview answer

A strong interview answer is:

> **Full fine-tuning updates all or nearly all parameters of a pre-trained language model. During backpropagation, gradients and optimizer states are maintained for the entire trainable model, which provides maximum adaptation capacity but requires significant compute, GPU memory, and storage.**
>
> **PEFT, or Parameter-Efficient Fine-Tuning, keeps most of the pre-trained model frozen and trains only a small number of parameters or additional modules. Examples include LoRA, QLoRA, adapter tuning, prompt tuning, and prefix tuning. PEFT reduces training cost and memory and makes it practical to maintain multiple task-specific adaptations using a shared base model.**

# Final memory trick

```text
FULL FINE-TUNING
────────────────────────

Base Model:
████████████████████

All weights:
✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓


PEFT
────────────────────────

Base Model:
████████████████████
❄️ ❄️ ❄️ ❄️ ❄️ ❄️

Small additional parameters:
✓ ✓
```

## The most important code difference

### Full fine-tuning

```python
optimizer = AdamW(
    model.parameters()
)
```

### PEFT

```python
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    )
)
```

And the key concept is:

```text
Full Fine-Tuning = Change the model itself

PEFT = Preserve the base model and learn a small adaptation layer
```

For your LLM interview preparation, remember this decision:

```text
Need current knowledge?
       → RAG

Need small behavior/task adaptation?
       → PEFT

Need deep model-wide adaptation and have sufficient data/compute?
       → Full Fine-Tuning
```
