# How does LoRA reduce GPU requirements?

LoRA (**Low-Rank Adaptation**) reduces GPU requirements mainly because it **freezes the large base model and trains only a small number of additional parameters**.

The core idea is:

```text
Full Fine-Tuning:
Train ALL 7 billion parameters

LoRA:
Freeze 7 billion base parameters
Train only small LoRA matrices
```

---

# 1. Why full fine-tuning needs so much GPU memory

During full fine-tuning:

```text
Trainable Base Model
        │
        ├── Model weights
        ├── Gradients for every weight
        ├── Optimizer states for every weight
        └── Activations
```

For a 7B model:

```text
7,000,000,000 trainable parameters
```

Conceptually:

```python
model = load_model()

for parameter in model.parameters():
    parameter.requires_grad = True
```

Then the optimizer receives **all parameters**:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)
```

AdamW creates optimizer state for every trainable parameter.

So:

```text
7B trainable parameters
       ↓
7B gradients
       ↓
7B Adam first moments
       ↓
7B Adam second moments
```

That is expensive.

---

# 2. What LoRA changes

Instead of changing the original weight matrix:

```text
W
```

LoRA freezes it:

```text
W = frozen
```

Then it learns a small update:

```text
ΔW
```

The effective weight becomes:

```text
W_new = W + ΔW
```

But LoRA does not directly store a full matrix for `ΔW`.

Instead:

```text
ΔW = B × A
```

where:

```text
A = small matrix
B = small matrix
```

---

# 3. Visual example

Suppose an original weight matrix is:

```text
W: 4096 × 4096
```

The number of parameters is:

```text
4096 × 4096
= 16,777,216 parameters
```

With LoRA rank:

```text
r = 16
```

Instead of training:

```text
4096 × 4096
= 16,777,216
```

we train:

```text
A: 16 × 4096
B: 4096 × 16
```

Total:

```text
16 × 4096
+
4096 × 16

= 131,072 parameters
```

Compare:

```text
Full matrix:
16,777,216 parameters

LoRA:
131,072 parameters
```

That's approximately:

```text
0.78%
```

of the original matrix.

---

# 4. Python calculation

```python
d = 4096
r = 16

full_parameters = d * d

lora_parameters = (
    r * d
    +
    d * r
)

reduction_percentage = (
    1
    -
    lora_parameters / full_parameters
) * 100


print(
    "Full parameters:",
    f"{full_parameters:,}"
)

print(
    "LoRA parameters:",
    f"{lora_parameters:,}"
)

print(
    "Reduction:",
    f"{reduction_percentage:.2f}%"
)
```

Output:

```text
Full parameters: 16,777,216
LoRA parameters: 131,072
Reduction: 99.22%
```

This is for **one weight matrix**.

A large LLM has many layers, and LoRA can be applied selectively.

---

# 5. LoRA architecture

Consider a normal transformer layer:

```text
Input X
    │
    ▼
    W
    │
    ▼
Output
```

With LoRA:

```text
                 ┌───────────── W (Frozen) ─────────────┐
                 │                                      │
Input X ─────────┼──────────────────────────────────────┼──► Output
                 │                                      │
                 │                                      │
                 └──► A ───► B ───► Scaling ────────────┘
                         Trainable
```

Mathematically:

```text
Output = XW + XBA
```

More commonly:

```text
W' = W + (α / r) × BA
```

Where:

* `W` = original frozen model weights
* `A` and `B` = trainable low-rank matrices
* `r` = LoRA rank
* `α` = scaling factor

---

# 6. How LoRA reduces gradient memory

This is one of the biggest benefits.

## Full fine-tuning

```python
for parameter in model.parameters():
    parameter.requires_grad = True
```

Every parameter needs gradients.

For a 7B model:

```text
7B parameters
        ↓
7B gradients
```

## LoRA

The base model is frozen:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

Then LoRA introduces small trainable matrices:

```text
Frozen base model

W1   W2   W3   W4
│    │    │    │
❌   ❌   ❌   ❌ gradients


LoRA adapters

A1/B1   A2/B2   A3/B3
  │       │       │
  ✓       ✓       ✓ gradients
```

So:

```text
Full Fine-Tuning:
7,000,000,000 gradients

LoRA:
Maybe 10M–100M gradients
```

The exact number depends on:

* LoRA rank
* Target modules
* Number of layers
* Model architecture

---

# 7. How LoRA reduces optimizer memory

This is often an even bigger saving.

Consider:

```python
optimizer = torch.optim.AdamW(
    trainable_parameters
)
```

AdamW stores approximately:

```text
m = first moment
v = second moment
```

For every trainable parameter:

```text
Parameter
    │
    ├── Gradient
    ├── Adam m
    └── Adam v
```

### Full fine-tuning

```text
7B trainable parameters
        │
        ├── 7B gradients
        ├── 7B m values
        └── 7B v values
```

### LoRA

```text
7B frozen parameters

+
20M trainable LoRA parameters
```

Therefore:

```text
Optimizer states are needed
only for the 20M parameters
```

This dramatically reduces memory.

---

# 8. Example: Calculate optimizer memory

Let's compare 7B full fine-tuning with 20M LoRA parameters.

```python
def gb(bytes_value):
    return bytes_value / (1024 ** 3)


full_parameters = 7_000_000_000
lora_parameters = 20_000_000


# FP32 Adam states:
# m + v
# 4 bytes each

def adam_memory(
    parameters
):

    m = parameters * 4

    v = parameters * 4

    return m + v


full_adam = adam_memory(
    full_parameters
)

lora_adam = adam_memory(
    lora_parameters
)


print(
    "Full fine-tuning Adam memory:",
    f"{gb(full_adam):.2f} GB"
)

print(
    "LoRA Adam memory:",
    f"{gb(lora_adam):.2f} GB"
)
```

Approximate result:

```text
Full fine-tuning Adam memory:
52.15 GB

LoRA Adam memory:
0.15 GB
```

That is a huge difference.

---

# 9. Complete memory comparison

Suppose a 7B model.

## Full fine-tuning

```text
Base model weights         ~14 GB
Gradients                  ~14 GB
Adam m                     ~26 GB
Adam v                     ~26 GB
Master weights*            ~26 GB
Activations                Variable
--------------------------------------
Total                      ~100+ GB
```

`*` Master-weight usage depends on the precision/training implementation.

---

## LoRA

```text
Base model weights         ~14 GB
LoRA parameters            Small
LoRA gradients             Small
LoRA optimizer states      Small
Activations                Variable
--------------------------------------
Total                      Much lower
```

For example:

```text
Base model                 ~14 GB
LoRA adapters              ~100 MB
Optimizer states           ~200 MB
Activations                ~3–15 GB
--------------------------------------
Total                      ~17–30+ GB
```

This varies significantly based on context length and batch size.

---

# 10. Actual LoRA implementation using PEFT

Install:

```bash
pip install transformers peft accelerate datasets
```

Load a model:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

MODEL_NAME = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Now configure LoRA:

```python
from peft import LoraConfig


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
```

Attach LoRA:

```python
from peft import (
    get_peft_model
)


model = get_peft_model(
    model,
    lora_config
)
```

Now check:

```python
model.print_trainable_parameters()
```

You might see:

```text
trainable params: 20,971,520

all params: 8,030,261,248

trainable%: 0.2612%
```

The exact output depends on the model and target modules.

---

# 11. What PEFT is doing internally

Conceptually, before LoRA:

```text
Linear Layer

Input
  │
  ▼
W
  │
  ▼
Output
```

During full training:

```text
W.requires_grad = True
```

With LoRA:

```text
Original W

requires_grad = False
```

And PEFT adds:

```text
         Frozen W
             │
Input ───────┼──────────►
             │
             │
        LoRA A + B
             │
             ▼
          Trainable
```

Conceptually:

```python
class SimpleLoRALayer(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=16
    ):
        super().__init__()

        # Frozen original layer

        self.weight = torch.nn.Parameter(
            torch.randn(
                out_features,
                in_features
            ),
            requires_grad=False
        )

        # Trainable LoRA matrices

        self.lora_A = torch.nn.Parameter(
            torch.randn(
                rank,
                in_features
            )
        )

        self.lora_B = torch.nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )

        self.rank = rank


    def forward(self, x):

        base_output = (
            x
            @
            self.weight.T
        )

        lora_output = (
            x
            @
            self.lora_A.T
            @
            self.lora_B.T
        )

        return (
            base_output
            +
            lora_output
        )
```

This is a simplified educational implementation. Production LoRA layers also handle scaling and initialization carefully.

---

# 12. Verify only LoRA parameters are trainable

You can inspect parameters:

```python
for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            "TRAINABLE:",
            name,
            parameter.numel()
        )
```

You will typically see:

```text
TRAINABLE:
q_proj.lora_A

TRAINABLE:
q_proj.lora_B

TRAINABLE:
v_proj.lora_A

TRAINABLE:
v_proj.lora_B
```

But the original weights are frozen.

---

# 13. Optimizer only receives trainable parameters

A good practice is:

```python
trainable_parameters = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
]
```

Then:

```python
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=2e-4
)
```

Check:

```python
total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Total: "
    f"{total_parameters:,}"
)

print(
    f"Trainable: "
    f"{trainable_parameters_count:,}"
)

print(
    "Trainable percentage:",
    (
        trainable_parameters_count
        /
        total_parameters
        *
        100
    )
)
```

This demonstrates exactly why optimizer memory is much lower.

---

# 14. What LoRA does NOT reduce much

This is important for interviews.

LoRA significantly reduces:

```text
✓ Trainable gradients
✓ Optimizer states
✓ Trainable parameter memory
```

But LoRA does **not eliminate**:

```text
✗ Base model memory
✗ Activation memory
✗ Attention computation
```

You still need to load the entire base model.

For example:

```text
7B model in BF16

Base model alone
≈ 13–14 GB
```

Therefore, LoRA alone may still be difficult on a small GPU.

---

# 15. Why QLoRA reduces memory further

LoRA:

```text
Base model → BF16/FP16
```

QLoRA:

```text
Base model → 4-bit
```

So:

```text
7B model

BF16
≈ 14 GB
```

becomes approximately:

```text
7B model

4-bit
≈ 3.5 GB raw parameter storage
```

with practical overhead above that.

Then:

```text
Frozen 4-bit Base Model
            +
Trainable BF16 LoRA Adapters
            =
QLoRA
```

Example:

```python
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Then:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

This is why QLoRA is often preferred when GPU memory is limited.

---

# 16. LoRA vs QLoRA

| Feature                  | Full Fine-Tuning | LoRA              | QLoRA                          |
| ------------------------ | ---------------- | ----------------- | ------------------------------ |
| Base model trainable     | Yes              | No                | No                             |
| Adapter trainable        | No               | Yes               | Yes                            |
| Gradients for base model | Yes              | No                | No                             |
| Optimizer for base model | Yes              | No                | No                             |
| Base model precision     | FP16/BF16        | Usually FP16/BF16 | 4-bit                          |
| GPU requirement          | Highest          | Lower             | Lowest                         |
| Training speed           | Can be high      | Often efficient   | Can have quantization overhead |

---

# 17. How much does LoRA actually save?

Consider:

```text
Full Fine-Tuning

7B trainable parameters
```

versus:

```text
LoRA

7B total parameters

20M trainable parameters
```

Trainable ratio:

```python
full = 7_000_000_000
lora = 20_000_000

percentage = (
    lora / full
) * 100

print(
    f"{percentage:.3f}%"
)
```

Output:

```text
0.286%
```

So only around:

```text
0.3%
```

of parameters might be trainable in this example.

That means:

```text
Full fine-tuning:
Huge gradients + optimizer states

LoRA:
Tiny fraction gets gradients + optimizer states
```

---

# 18. Complete training example

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import SFTTrainer
```

Load model:

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token
```

LoRA configuration:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
```

Attach:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Enable memory optimization:

```python
model.gradient_checkpointing_enable()

model.config.use_cache = False
```

Training arguments:

```python
training_args = TrainingArguments(
    output_dir="./lora-output",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    bf16=True,

    logging_steps=10,

    save_steps=100,

    gradient_checkpointing=True
)
```

The important part is that:

```text
7B base parameters → frozen

LoRA parameters → trainable
```

---

# 19. Interview-ready answer

> **LoRA reduces GPU requirements by freezing the original model weights and training only small low-rank adapter matrices. During full fine-tuning, every model parameter needs gradients and optimizer states, which are extremely expensive for a billion-parameter model.**
>
> **With LoRA, a weight update is represented as a low-rank decomposition, ΔW = BA, instead of training the entire weight matrix. This means gradients and optimizer states are required only for the small LoRA matrices rather than all 7 billion parameters.**
>
> **LoRA does not eliminate the memory required to load the base model or store activations. That's why QLoRA goes further by loading the frozen base model in 4-bit quantization while training LoRA adapters in higher precision. I would combine LoRA with gradient checkpointing, small micro-batches, and gradient accumulation to fit training on limited GPU memory.**

# The easiest way to remember it

```text
FULL FINE-TUNING

7B parameters
   ↓
Train everything
   ↓
Gradients for everything
   ↓
Optimizer states for everything
   ↓
Very high GPU memory
```

```text
LORA

7B base parameters
   ↓
Freeze everything
   +
Small A/B matrices
   ↓
Train only adapters
   ↓
Small gradients
   ↓
Small optimizer states
   ↓
Much lower GPU memory
```

**LoRA's biggest memory saving is not that the 7B model disappears—it is that you no longer maintain training state for all 7 billion base-model parameters.**
