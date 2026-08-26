# Configure `LoraConfig` for LoRA / QLoRA

`LoraConfig` tells PEFT:

* **which layers** should receive LoRA adapters
* the **LoRA rank** (`r`)
* the **scaling factor** (`lora_alpha`)
* the **dropout**
* the **task type**
* whether bias parameters are trained

The basic flow is:

```text
Pretrained LLM
     │
     ▼
 LoraConfig
     │
     ▼
Select target modules
     │
     ▼
Insert LoRA adapters
     │
     ▼
Freeze base model
     │
     ▼
Train LoRA weights only
```

---

## 1. Basic `LoraConfig`

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

This is a common starting configuration for LLM fine-tuning.

---

# 2. What each parameter means

## `r` — LoRA rank

```python
r=16
```

LoRA approximates a weight update:

$$
\Delta W = BA
$$

Instead of training:

$$
W \in \mathbb{R}^{d \times d}
$$

we train two smaller matrices:

$$
A \in \mathbb{R}^{r \times d}
$$

$$
B \in \mathbb{R}^{d \times r}
$$

For example:

```text
Original matrix

4096 × 4096
```

With:

```python
r=16
```

LoRA trains approximately:

```text
A = 16 × 4096
B = 4096 × 16
```

### Rank tradeoff

| Rank | Parameters | Capacity   | Memory   |
| ---- | ---------: | ---------- | -------- |
| 4    |        Low | Low        | Very low |
| 8    |        Low | Medium-low | Low      |
| 16   |     Medium | Good       | Low      |
| 32   |     Higher | High       | Medium   |
| 64   |       High | Very high  | Higher   |

A good starting point:

```python
r=8
```

or:

```python
r=16
```

For a difficult domain:

```python
r=32
```

---

# 3. `lora_alpha` — LoRA scaling

```python
lora_alpha=32
```

LoRA scaling is:

$$
\text{Scaling} = \frac{\alpha}{r}
$$

Therefore:

```python
r = 16
lora_alpha = 32
```

gives:

$$
\frac{32}{16} = 2
$$

The effective weight becomes:

$$
W' = W + \frac{\alpha}{r}BA
$$

Code:

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

A common relationship is:

```python
r=8
lora_alpha=16
```

or:

```python
r=16
lora_alpha=32
```

---

# 4. `lora_dropout`

```python
lora_dropout=0.05
```

Dropout is applied during training to help reduce overfitting.

Conceptually:

```text
LoRA Input
    │
    ▼
Dropout
    │
    ▼
LoRA A
    │
    ▼
LoRA B
```

Typical values:

```text
0.0   → Large dataset / less regularization
0.05  → Good common starting point
0.1   → Small dataset / higher overfitting risk
```

Example:

```python
lora_dropout=0.05
```

---

# 5. `target_modules`

This is one of the most important settings.

Consider a transformer attention block:

```text
Input
  │
  ├── q_proj → Query
  │
  ├── k_proj → Key
  │
  ├── v_proj → Value
  │
  └── o_proj → Output
```

A common configuration:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

This applies LoRA to all major attention projections.

---

## Why target attention projections?

Attention controls:

```text
What information to look at
Which tokens are important
How information is combined
```

LoRA changes the transformation behavior without modifying the original base weights.

Mathematically:

```text
Original:

Y = XW
```

With LoRA:

$$
Y = X(W + \Delta W)
$$

where:

$$
\Delta W = BA
$$

So the model behavior changes while:

```text
Base W → Frozen
LoRA A → Trainable
LoRA B → Trainable
```

---

# 6. Attention-only configuration

A common efficient configuration:

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

This is a good default for many decoder-only models.

---

# 7. Attention + MLP configuration

For more adaptation capacity:

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
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

Architecture:

```text
Transformer Block

          Input
            │
      ┌─────▼─────┐
      │ Attention │
      │ q,k,v,o   │ ← LoRA
      └─────┬─────┘
            │
      ┌─────▼─────┐
      │    MLP    │
      │ gate/up/  │ ← LoRA
      │ down      │
      └─────┬─────┘
            │
          Output
```

### Advantages

```text
More learning capacity
Better domain adaptation
More flexibility
```

### Disadvantages

```text
More trainable parameters
More GPU memory
Slower training
```

---

# 8. `bias`

Most commonly:

```python
bias="none"
```

This means:

```text
Base model biases → Frozen
LoRA matrices → Trainable
```

Other options can include:

```python
bias="all"
```

or:

```python
bias="lora_only"
```

But usually:

```python
bias="none"
```

is a good default because it preserves parameter efficiency.

---

# 9. `task_type`

For a decoder-only LLM:

```python
task_type="CAUSAL_LM"
```

Examples:

```text
Llama
Qwen
Mistral
Gemma
```

Other tasks can use different types.

Conceptually:

```text
Causal LM
```

means the model predicts:

```text
Next token
```

Example:

```text
Input:
The capital of France is

Target:
Paris
```

For instruction fine-tuning, chat fine-tuning, and QLoRA on decoder LLMs:

```python
task_type="CAUSAL_LM"
```

is typically correct.

---

# 10. Complete configuration for QLoRA

```python
from peft import LoraConfig


lora_config = LoraConfig(

    # Low-rank dimension
    r=16,

    # Scaling
    lora_alpha=32,

    # Regularization
    lora_dropout=0.05,

    # Layers to adapt
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    # Don't train bias
    bias="none",

    # Decoder-only LLM
    task_type="CAUSAL_LM"
)
```

Then:

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)
```

---

# 11. Verify which parameters are trainable

Always check:

```python
model.print_trainable_parameters()
```

Example:

```text
trainable params: 8,388,608
all params: 1,500,000,000
trainable%: 0.56%
```

This is the major benefit of LoRA.

```text
1.5 Billion Parameters
        │
        ▼
Only 8 Million Trainable
```

If you unexpectedly see:

```text
trainable params: 1,500,000,000
```

then your LoRA configuration is probably wrong because the full model may be trainable.

---

# 12. Inspect target modules before configuring LoRA

Different model architectures use different layer names.

For example, one model might use:

```text
q_proj
k_proj
v_proj
o_proj
```

Another might use:

```text
query
key
value
dense
```

Inspect the model:

```python
for name, module in model.named_modules():

    if "proj" in name:
        print(name)
```

For more detail:

```python
for name, module in model.named_modules():

    print(
        name,
        type(module)
    )
```

Example output:

```text
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj
```

Then use:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

This step is important because incorrect module names can cause errors or result in LoRA being applied to the wrong layers.

---

# 13. Different configurations for different scenarios

## Small dataset

```python
small_dataset_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

Why?

```text
Smaller rank
+
Higher dropout
=
Lower overfitting risk
```

---

## Medium dataset

```python
medium_dataset_config = LoraConfig(
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

This is a strong default.

---

## Large domain-specific dataset

```python
large_dataset_config = LoraConfig(
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

This gives more adaptation capacity.

---

# 14. Complete QLoRA integration

This is how `LoraConfig` fits into the full pipeline:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ---------------------------------------------
# 1. Configure 4-bit quantization
# ---------------------------------------------

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ---------------------------------------------
# 2. Load quantized model
# ---------------------------------------------

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)


# ---------------------------------------------
# 3. Prepare for QLoRA training
# ---------------------------------------------

model = prepare_model_for_kbit_training(
    model
)


# ---------------------------------------------
# 4. Configure LoRA
# ---------------------------------------------

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


# ---------------------------------------------
# 5. Attach adapters
# ---------------------------------------------

model = get_peft_model(
    model,
    lora_config
)


# ---------------------------------------------
# 6. Verify
# ---------------------------------------------

model.print_trainable_parameters()
```

The result:

```text
                    QLoRA Model

           ┌──────────────────────┐
           │   Base LLM           │
           │                      │
           │  4-bit NF4           │
           │  FROZEN              │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │   LoRA Adapters      │
           │                      │
           │   r = 16             │
           │   alpha = 32         │
           │   dropout = 0.05     │
           │                      │
           │   TRAINABLE ✓        │
           └──────────────────────┘
```

---

# 15. How I would choose `LoraConfig` in a real project

I would not blindly choose:

```python
r=16
```

Instead, I would experiment with:

```text
Experiment 1:
r=8

Experiment 2:
r=16

Experiment 3:
r=32
```

And compare:

```text
Validation Loss
Task Accuracy
Hallucination Rate
Domain-specific Evaluation
Response Quality
Latency
GPU Memory
```

Example configuration:

```python
configs = [

    {"r": 8, "alpha": 16},

    {"r": 16, "alpha": 32},

    {"r": 32, "alpha": 64}
]
```

Choose the **smallest rank that gives acceptable validation performance**.

---

# Interview-ready answer

> **`LoraConfig` defines how LoRA adapters are attached to a pretrained model. The most important parameters are the rank `r`, which controls adapter capacity, `lora_alpha`, which controls the scaling of the LoRA update, `lora_dropout` for regularization, and `target_modules`, which specifies which transformer layers receive adapters.**
>
> **For a decoder-only LLM, I typically use `task_type="CAUSAL_LM"` and start with `r=8` or `r=16`. For target modules, I commonly start with attention projections such as `q_proj` and `v_proj`, then potentially include `k_proj`, `o_proj`, and MLP projections depending on the adaptation task and available GPU memory. I verify the architecture first because module names vary across models. Finally, I compare configurations using validation metrics and choose the smallest configuration that achieves the required quality.**

A solid default for your QLoRA project is:

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

This is the `LoraConfig` that you can directly plug into the QLoRA training pipeline we built.
