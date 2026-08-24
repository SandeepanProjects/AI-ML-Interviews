# QLoRA vs LoRA — Difference with Code

The simplest answer is:

> **LoRA reduces the number of trainable parameters. QLoRA reduces both trainable parameters and base-model memory.**

---

# 1. The core difference

## LoRA

```text
Base Model
FP16 / BF16
Frozen ❄️

+
LoRA Adapters
Trainable ✓
```

## QLoRA

```text
Base Model
4-bit Quantized
Frozen ❄️

+
LoRA Adapters
Trainable ✓
```

### Main difference

| Feature                   | LoRA              | QLoRA                                  |
| ------------------------- | ----------------- | -------------------------------------- |
| Base model storage        | Usually FP16/BF16 | Usually 4-bit                          |
| Base model trainable      | ❌ Frozen          | ❌ Frozen                               |
| LoRA adapters             | ✅ Trainable       | ✅ Trainable                            |
| Gradients for base model  | ❌                 | ❌                                      |
| Optimizer states for base | ❌                 | ❌                                      |
| GPU memory                | Lower             | Even lower                             |
| Training complexity       | Simpler           | More complex                           |
| Quantization              | Not required      | Core technique                         |
| Best for                  | Enough GPU memory | Limited GPU memory / very large models |

---

# 2. First understand LoRA

A normal linear layer has:

[
Y = XW
]

where (W) is the pretrained weight matrix.

LoRA freezes (W_0) and learns a small update:

[
W = W_0 + \Delta W
]

Instead of training the full matrix:

[
\Delta W \in \mathbb{R}^{d \times k}
]

LoRA approximates it as:

[
\Delta W = BA
]

where:

[
A \in \mathbb{R}^{r \times k}
]

[
B \in \mathbb{R}^{d \times r}
]

and:

[
r \ll d,k
]

So:

[
\boxed{
W = W_0 + \frac{\alpha}{r}BA
}
]

Only (A) and (B) are trained.

---

# 3. LoRA architecture

```text
                     Input X
                        │
                        ▼
                ┌───────────────┐
                │ Base Weight W₀│
                │ Frozen ❄️     │
                └───────┬───────┘
                        │
                   Base Output
                        │
                        │
         ┌──────────────┼─────────────┐
         │                            │
         ▼                            ▼
    X × W₀                    X × A × B
                              LoRA path
                                │
                         × (α / r)
                                │
         └──────────────┬─────────────┘
                        ▼
                     Output
```

The base model remains frozen, while LoRA learns the small low-rank update.

---

# 4. LoRA code

## Step 1: Load the model

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

MODEL_NAME = "meta-llama/Llama-3.1-8B"
```

Load normally using BF16:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

At this point:

```text
Base Model
8B parameters
BF16
```

The approximate raw weight memory is:

[
8B \times 2 \text{ bytes}
\approx 16GB
]

before runtime overhead.

---

## Step 2: Configure LoRA

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

    task_type=TaskType.CAUSAL_LM
)
```

---

## Step 3: Add adapters

```python
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Conceptually:

```text
Base Model
BF16
Frozen ❄️

       +

LoRA A and B
Trainable ✓
```

Only the adapters receive gradients and optimizer states.

---

# 5. What QLoRA changes

QLoRA keeps the same LoRA concept:

[
W = W_0 + \frac{\alpha}{r}BA
]

But it changes how the base model is stored.

Instead of:

```text
W₀
BF16
```

QLoRA uses:

```text
W₀
4-bit quantized
```

So:

```text
LoRA:

BF16 Frozen Base
+
Trainable LoRA


QLoRA:

4-bit Frozen Base
+
Trainable LoRA
```

---

# 6. QLoRA code

The major difference appears when loading the base model.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model
)
```

---

## Step 1: Configure 4-bit quantization

```python
bnb_config = BitsAndBytesConfig(

    # Load base model in 4-bit
    load_in_4bit=True,

    # Quantization format
    bnb_4bit_quant_type="nf4",

    # Double quantization
    bnb_4bit_use_double_quant=True,

    # Computation precision
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

## Step 2: Load the model in 4-bit

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Now:

```text
Base Model
8B parameters
4-bit
Frozen later
```

Approximate raw weight storage:

[
8B \times 0.5
=============

4GB
]

plus quantization metadata and runtime memory.

Compare with LoRA:

```text
LoRA Base Model:

8B × 2 bytes
≈ 16 GB


QLoRA Base Model:

8B × 0.5 bytes
≈ 4 GB raw storage
```

---

# 7. Step unique to QLoRA

QLoRA typically prepares the quantized model for training:

```python
model = prepare_model_for_kbit_training(
    model
)
```

This is a major difference.

### LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)
```

### QLoRA:

```python
model = prepare_model_for_kbit_training(
    model
)

model = get_peft_model(
    model,
    lora_config
)
```

---

# 8. Add the same LoRA adapters

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

    task_type=TaskType.CAUSAL_LM
)
```

Apply:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Final architecture:

```text
        QLoRA Model

┌────────────────────────────┐
│                            │
│  Base LLM                  │
│                            │
│  4-bit Quantized           │
│  Frozen ❄️                 │
│                            │
└──────────────┬─────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Base Output      LoRA A → B
                    Trainable ✓
       │                │
       └───────┬────────┘
               ▼
             Output
```

---

# 9. Complete side-by-side code

## LoRA

```python
import torch

from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


MODEL_NAME = "your-model"


# ---------------------------------
# Load normal BF16 model
# ---------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


# ---------------------------------
# LoRA configuration
# ---------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type=TaskType.CAUSAL_LM
)


# ---------------------------------
# Add LoRA adapters
# ---------------------------------

model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()
```

---

## QLoRA

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model
)


MODEL_NAME = "your-model"


# ---------------------------------
# 4-bit configuration
# ---------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)


# ---------------------------------
# Load 4-bit model
# ---------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# ---------------------------------
# Prepare quantized model
# ---------------------------------

model = prepare_model_for_kbit_training(
    model
)


# ---------------------------------
# LoRA configuration
# ---------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type=TaskType.CAUSAL_LM
)


# ---------------------------------
# Add LoRA adapters
# ---------------------------------

model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()
```

---

# 10. The key code difference

This is the easiest way to remember it.

## LoRA

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

model = get_peft_model(
    model,
    lora_config
)
```

## QLoRA

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config
)

model = prepare_model_for_kbit_training(
    model
)

model = get_peft_model(
    model,
    lora_config
)
```

### Therefore:

```text
LoRA
=
Frozen normal-precision base
+
Train LoRA


QLoRA
=
Frozen quantized base
+
Train LoRA
```

---

# 11. Memory comparison

Suppose we have a 70B model.

## LoRA

Base model in BF16:

[
70B \times 2
============

140GB
]

```text
140 GB base weights
+
Adapters
+
Activations
+
Temporary buffers
```

This generally requires multiple GPUs or very large GPU memory.

---

## QLoRA

Base model in 4-bit:

[
70B \times 0.5
==============

35GB
]

```text
35 GB raw 4-bit weights
+
Quantization metadata
+
LoRA adapters
+
Activations
+
Temporary buffers
```

Still a very large workload, but much more feasible.

### Important

You should **not** say:

> "A 70B QLoRA model always needs exactly 35 GB."

Because 35 GB only represents approximate raw 4-bit parameter storage. Actual training memory depends on:

* sequence length
* batch size
* activation memory
* CUDA workspace
* model architecture
* quantization metadata
* LoRA target modules
* gradient checkpointing

---

# 12. Training difference

## LoRA training

```text
Input
  │
  ▼
BF16 Base Model
Frozen ❄️
  │
  ▼
LoRA
Trainable ✓
  │
  ▼
Loss
  │
  ▼
Update LoRA
```

## QLoRA training

```text
Input
  │
  ▼
4-bit Quantized Base Model
Frozen ❄️
  │
  ▼
LoRA
Trainable ✓
  │
  ▼
Loss
  │
  ▼
Update LoRA
```

The **training algorithm for the adapters is fundamentally similar**.

The major difference is the memory-efficient representation of the frozen base model.

---

# 13. Why doesn't QLoRA train 4-bit weights?

A common misconception is:

```text
QLoRA = Train model in 4-bit
```

That is not the best way to think about it.

Instead:

```text
QLoRA

4-bit Base Model
       ❄️ Frozen

       +

Higher-precision LoRA adapters
       ✓ Trainable
```

The gradients are used to update the LoRA parameters.

The base model stays frozen.

---

# 14. Real-world example

Imagine you are building an:

```text
Enterprise Financial AI Assistant
```

You want to fine-tune an 8B model.

## You have a large GPU cluster

You could use:

```text
BF16 Base Model
+
LoRA
```

This gives:

```text
Simpler setup
Potentially fewer quantization-related issues
```

---

## You only have limited GPU memory

Use:

```text
4-bit Base Model
+
LoRA
```

This gives:

```text
Much lower memory usage
```

This is QLoRA.

---

# 15. When should you use LoRA?

Use LoRA when:

```text
✓ You have enough GPU memory
✓ You want a simpler training setup
✓ You want to avoid aggressive base-weight quantization
✓ You want PEFT but don't need maximum memory savings
✓ Your model already fits comfortably in BF16/FP16
```

Example:

```text
8B model
on a 48GB GPU
```

Normal LoRA may be practical.

---

# 16. When should you use QLoRA?

Use QLoRA when:

```text
✓ GPU memory is limited
✓ The model is very large
✓ You want to fine-tune a 30B/70B-class model
✓ You want lower training cost
✓ You are experimenting with many adapters
```

Example:

```text
70B model
+
limited GPU resources
```

QLoRA is usually much more practical than BF16 LoRA.

---

# 17. Important quality trade-off

```text
LoRA:
Higher precision base weights
        ↓
Potentially more faithful to original weights


QLoRA:
Quantized base weights
        ↓
Small quantization approximation error
```

However, QLoRA was specifically designed so that 4-bit quantization can still achieve strong fine-tuning results. The quality difference depends on:

* model
* dataset
* quantization implementation
* training hyperparameters
* target modules
* task

You should evaluate on your actual task rather than assuming one is always better.

---

# 18. Full comparison

| Feature               | LoRA                          | QLoRA                         |
| --------------------- | ----------------------------- | ----------------------------- |
| Technique             | Low-rank adaptation           | Quantized low-rank adaptation |
| Base precision        | Usually FP16/BF16             | Usually 4-bit                 |
| Base weights          | Frozen                        | Frozen                        |
| Trainable parameters  | LoRA adapters                 | LoRA adapters                 |
| Base gradients        | No                            | No                            |
| Base optimizer states | No                            | No                            |
| Quantization          | Optional                      | Essential                     |
| Memory usage          | Low                           | Much lower                    |
| Complexity            | Lower                         | Higher                        |
| Best for              | Enough GPU memory             | Memory-constrained training   |
| Large models          | Possible                      | More practical                |
| 70B models            | Difficult on limited hardware | Much more feasible            |

---

# 19. The most important conceptual difference

Think of it this way:

```text
FULL FINE-TUNING

Normal precision base
+
Train everything


LoRA

Normal precision base
Frozen
+
Train small adapters


QLoRA

4-bit compressed base
Frozen
+
Train small adapters
```

---

# Best interview answer

> **LoRA and QLoRA both use low-rank adapters and freeze the original pretrained model. The main difference is how the base model is stored. In standard LoRA, the frozen base model is typically loaded in BF16 or FP16, while in QLoRA it is loaded in 4-bit quantized form, usually using NF4. Both train only the LoRA adapter parameters, but QLoRA significantly reduces GPU memory because the billions of frozen base parameters are stored in 4-bit instead of 16-bit precision. I would use LoRA when the model comfortably fits in GPU memory, and QLoRA when memory is constrained or I need to fine-tune very large models such as 70B-class models.**

## One-line memory trick

[
\boxed{
\text{LoRA} = \text{Frozen Model} + \text{Train Adapters}
}
]

[
\boxed{
\text{QLoRA} =
\text{4-bit Frozen Model}
+
\text{Train Adapters}
}
]
