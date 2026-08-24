# What precision are LoRA adapters trained in?

The important point is:

> **In QLoRA, the base model is stored in 4-bit, but the LoRA adapter parameters are normally trained in FP16 or BF16 — not 4-bit.**

For modern NVIDIA GPUs, **BF16 is often preferred** when supported.

So the typical QLoRA setup is:

```text
┌─────────────────────────────┐
│ Base LLM                    │
│                             │
│ Stored: 4-bit NF4           │
│ Frozen ❄️                   │
└──────────────┬──────────────┘
               │
               │ computation
               ▼
          BF16/FP16
               │
               ▼
┌─────────────────────────────┐
│ LoRA Adapter                │
│                             │
│ Parameters: BF16/FP16       │
│ Trainable ✓                 │
└─────────────────────────────┘
```

Let's understand exactly what happens.

---

# 1. There are several different precisions

When people say:

> "What precision is QLoRA using?"

they may actually be referring to **three different things**:

| Component            | Typical precision                           |
| -------------------- | ------------------------------------------- |
| Base model storage   | 4-bit NF4                                   |
| Computation          | BF16 or FP16                                |
| LoRA adapter weights | BF16 or FP16                                |
| LoRA gradients       | Usually same/appropriate training precision |
| Optimizer states     | Often FP32 or 8-bit/paged variant           |

This distinction is extremely important in interviews.

---

# 2. Base model: 4-bit

Suppose we configure:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

This means:

```text
Base model weights
        ↓
Stored in 4-bit NF4
```

But:

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

means computations involving the quantized weights use BF16 computation where applicable.

So don't interpret:

```python
load_in_4bit=True
```

as:

> "Everything in the model is now 4-bit."

That's incorrect.

---

# 3. LoRA adapters are not 4-bit

Suppose we have a linear layer:

[
W_0
]

LoRA introduces:

[
\Delta W = BA
]

and:

[
W_{\text{effective}}
====================

W_0 + \frac{\alpha}{r}BA
]

The base:

```text
W₀
↓
4-bit
Frozen
```

The LoRA matrices:

```text
A
B
↓
BF16/FP16
Trainable
```

So:

```text
4-bit Base
    +
BF16 LoRA
```

is a good mental model for QLoRA.

---

# 4. Code: create LoRA adapters

Let's use a realistic configuration.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
)
```

Configure the 4-bit base model:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Load:

```python
model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=bnb_config,
    device_map="auto",
)
```

Prepare:

```python
model = prepare_model_for_kbit_training(
    model
)
```

Now create LoRA:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type=TaskType.CAUSAL_LM,
)
```

Add it:

```python
model = get_peft_model(
    model,
    lora_config,
)
```

---

# 5. Check the actual LoRA dtype

This is one of the most useful pieces of code for understanding what is happening.

```python
for name, param in model.named_parameters():

    if "lora_" in name:
        print(
            name,
            param.dtype,
            param.requires_grad
        )
```

Depending on the exact Transformers/PEFT configuration and model, you may see something like:

```text
...q_proj.lora_A.default.weight
torch.float32
True

...q_proj.lora_B.default.weight
torch.float32
True
```

This surprises many people.

Why?

Because **creating the LoRA layers does not automatically guarantee that their stored parameter dtype is BF16**.

The exact dtype can depend on the model-loading/training setup.

---

# 6. Why can LoRA parameters initially be FP32?

Many PEFT workflows initially create trainable adapter parameters in FP32 for numerical stability.

You can explicitly inspect:

```python
for name, param in model.named_parameters():

    if param.requires_grad:
        print(name, param.dtype)
```

You might find:

```text
LoRA parameters → float32
```

even though you're intending to train using BF16.

This is why we need to distinguish:

> **parameter storage dtype**

from:

> **training computation dtype**

and:

> **mixed-precision training configuration**.

---

# 7. BF16 training

Suppose your GPU supports BF16.

You can configure:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    bf16=True,

    fp16=False,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,
)
```

Now training uses mixed precision with BF16 where supported.

Conceptually:

```text
Forward
   ↓
BF16 computation
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Optimizer
   ↓
LoRA update
```

---

# 8. FP16 alternative

If your GPU doesn't support BF16 well:

```python
training_args = TrainingArguments(
    output_dir="./output",

    bf16=False,

    fp16=True,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,
)
```

Then:

```text
LoRA training
       ↓
FP16 mixed precision
```

---

# 9. BF16 vs FP16

Both use:

[
16\text{ bits}
]

but they represent numbers differently.

### FP16

```text
1 sign bit
5 exponent bits
10 fraction bits
```

### BF16

```text
1 sign bit
8 exponent bits
7 fraction bits
```

Therefore:

```text
FP16
More mantissa precision
Smaller numerical range


BF16
Less mantissa precision
Much larger numerical range
```

For LLM training, BF16 is often attractive because it has the same exponent range as FP32.

---

# 10. Why BF16 is often preferred for LLM training

Suppose your GPU supports BF16.

Then:

```python
bf16=True
```

is often preferable to:

```python
fp16=True
```

because BF16 has better numerical range.

Conceptually:

```text
FP16
     ↓
Can overflow more easily


BF16
     ↓
Much larger exponent range
     ↓
Often more stable for large-model training
```

This is one reason modern LLM training stacks frequently use BF16.

---

# 11. What about the LoRA weights themselves?

You can explicitly convert LoRA parameters to BF16 if your training setup calls for it.

For example:

```python
for name, param in model.named_parameters():

    if "lora_" in name:
        param.data = param.data.to(
            torch.bfloat16
        )
```

Then inspect:

```python
for name, param in model.named_parameters():

    if "lora_" in name:
        print(
            name,
            param.dtype
        )
```

You should see:

```text
torch.bfloat16
```

However, **don't blindly add this to every QLoRA implementation**. Let the Transformers/PEFT mixed-precision setup handle dtype management unless you have a specific reason to force adapter storage to BF16.

---

# 12. A better way to inspect the model

Create a helper:

```python
def inspect_dtypes(model):

    for name, param in model.named_parameters():

        if "lora_" in name:
            print(
                "LoRA:",
                name,
                "| dtype:",
                param.dtype,
                "| trainable:",
                param.requires_grad,
            )
```

Run:

```python
inspect_dtypes(model)
```

You might get:

```text
LoRA:
...q_proj.lora_A.default.weight
| dtype: torch.float32
| trainable: True

LoRA:
...q_proj.lora_B.default.weight
| dtype: torch.float32
| trainable: True
```

or BF16 depending on how your model and training stack were configured.

The important thing is that:

```text
requires_grad=True
```

for LoRA.

---

# 13. What does `bf16=True` actually do?

This is another common interview question.

When you write:

```python
TrainingArguments(
    bf16=True
)
```

you are enabling mixed-precision BF16 training.

It does **not necessarily mean every parameter is permanently stored as BF16**.

Instead, the framework uses BF16 for supported forward/backward computations while managing parameter/optimizer states appropriately.

Think:

```text
Parameter representation
        ≠
Computation precision
        ≠
Optimizer state precision
```

---

# 14. What happens during a QLoRA forward pass?

Let's simplify the process.

Suppose:

```text
Base weight
W₀
```

is stored:

```text
4-bit NF4
```

During computation:

```text
4-bit W₀
    │
    ▼
Dequantization / computation path
    │
    ▼
BF16
```

LoRA:

```text
A
B
```

is used in higher precision.

Conceptually:

[
Y =
XW_0 +
XBA
]

where:

```text
W₀ → quantized base
BA → LoRA update
```

The final computation occurs using appropriate higher-precision arithmetic.

---

# 15. What happens during backward?

```text
Input
  │
  ▼
4-bit Base
  │
  │ frozen
  ▼
BF16 computation
  │
  ├───────────────┐
  │               │
  ▼               ▼
Base path       LoRA path
                 │
              BF16/FP16
                 │
                 ▼
               Output
                 │
                 ▼
                Loss
                 │
                 ▼
            Backpropagation
                 │
                 ▼
          LoRA gradients
                 │
                 ▼
          Optimizer update
```

The base model isn't updated.

---

# 16. What about optimizer precision?

This is another level of precision.

Suppose you use:

```python
optim="paged_adamw_32bit"
```

Then optimizer states can use 32-bit precision.

So you can have:

```text
Base weights
    ↓
4-bit

LoRA computation
    ↓
BF16

LoRA parameters
    ↓
BF16 or framework-managed dtype

Optimizer states
    ↓
FP32
```

This is completely valid.

There is no requirement that every component have the same precision.

---

# 17. Complete QLoRA training example

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
)


MODEL_NAME = "your-model"


# ==========================================
# 1. Quantization configuration
# ==========================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    # Computation dtype
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ==========================================
# 2. Load base model
# ==========================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",
)


# ==========================================
# 3. Prepare for k-bit training
# ==========================================

model = prepare_model_for_kbit_training(
    model
)


# ==========================================
# 4. LoRA configuration
# ==========================================

lora_config = LoraConfig(
    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type=TaskType.CAUSAL_LM,
)


# ==========================================
# 5. Add LoRA
# ==========================================

model = get_peft_model(
    model,
    lora_config
)


# ==========================================
# 6. Inspect trainable parameters
# ==========================================

model.print_trainable_parameters()


# ==========================================
# 7. Inspect LoRA dtypes
# ==========================================

for name, param in model.named_parameters():

    if "lora_" in name:

        print(
            name,
            param.dtype,
            param.requires_grad
        )


# ==========================================
# 8. Training configuration
# ==========================================

training_args = TrainingArguments(

    output_dir="./qlora-output",

    # BF16 mixed precision
    bf16=True,

    fp16=False,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    gradient_checkpointing=True,

    learning_rate=2e-4,

    optim="paged_adamw_32bit",

    num_train_epochs=3,

    logging_steps=10,

    save_steps=500,

    report_to="none",
)
```

---

# 18. So what precision should you choose?

### If your GPU supports BF16:

I would generally start with:

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

and:

```python
TrainingArguments(
    bf16=True,
    fp16=False,
)
```

So your conceptual configuration is:

```text
Base model
    ↓
4-bit NF4 storage

Computation
    ↓
BF16

LoRA training
    ↓
BF16/mixed precision

Optimizer
    ↓
Often FP32 or 8-bit/paged variant
```

---

# 19. LoRA vs QLoRA precision

This makes the difference very clear:

### LoRA

```text
Base model
    ↓
BF16/FP16
Frozen

LoRA
    ↓
BF16/FP16 / framework-managed
Trainable
```

### QLoRA

```text
Base model
    ↓
4-bit NF4
Frozen

LoRA
    ↓
BF16/FP16 / framework-managed
Trainable
```

So **QLoRA does not mean the LoRA adapter itself is 4-bit.**

That's the key point.

---

# 20. Interview answer

If the interviewer asks:

> **"What precision are LoRA adapters trained in during QLoRA?"**

A strong answer is:

> **"The quantized base model is typically stored in 4-bit NF4, but the LoRA adapters are trained using higher-precision arithmetic, typically BF16 or FP16. On modern GPUs that support it, I would generally prefer BF16. The exact parameter storage dtype can be framework-dependent, so I distinguish adapter parameter dtype from computation dtype. The optimizer states may also use FP32 or an 8-bit/paged optimizer. The important point is that QLoRA does not train the LoRA adapters in 4-bit just because the base model is quantized to 4-bit."**

### Remember:

[
\boxed{
\text{Base} = 4\text{-bit}
}
]

[
\boxed{
\text{LoRA} = \text{BF16/FP16 training}
}
]

[
\boxed{
\text{Computation} = \text{usually BF16/FP16}
}
]

[
\boxed{
\text{Optimizer} = \text{often FP32 or memory-efficient variant}
}
]

That's the precision picture you should be able to explain in an LLM/GenAI interview.
