# LoRA `r`, `alpha`, and `dropout` + Which layers to apply LoRA to

These are some of the most common **LLM fine-tuning interview questions**.

---

# 1. Quick overview

A LoRA layer modifies a frozen weight matrix:

[
W' = W + \Delta W
]

Instead of learning the complete update (\Delta W), LoRA learns:

[
\Delta W = BA
]

The actual LoRA output is commonly represented as:

[
Y = XW + \frac{\alpha}{r}XBA
]

where:

* `r` = LoRA rank
* `alpha` = scaling factor
* `dropout` = regularization applied to the LoRA path

---

# 2. What is LoRA rank `r`?

## Definition

`r` determines the size of the low-rank matrices.

Suppose the original weight matrix is:

[
W \in \mathbb{R}^{d_{out} \times d_{in}}
]

LoRA introduces:

[
A \in \mathbb{R}^{r \times d_{in}}
]

[
B \in \mathbb{R}^{d_{out} \times r}
]

Therefore:

[
\Delta W = BA
]

---

## Visual example

Suppose:

```text
Input dimension  = 4096
Output dimension = 4096
```

Original matrix:

```text
W = 4096 × 4096
```

Full fine-tuning would update:

[
4096 \times 4096 = 16,777,216
]

parameters.

Now choose:

```text
r = 8
```

LoRA matrices:

```text
Input X
  │
  ▼
A: 4096 → 8
  │
  ▼
Low-rank space
  │
  ▼
B: 8 → 4096
```

Trainable parameters:

[
(4096 \times 8) + (8 \times 4096)
]

[
= 65,536
]

So:

```text
Full fine-tuning:
16,777,216 parameters

LoRA r=8:
65,536 parameters
```

---

## Why does a higher rank increase capacity?

Compare:

```text
r = 2
```

```text
4096 → 2 → 4096
```

Very limited adaptation capacity.

```text
r = 8
```

```text
4096 → 8 → 4096
```

Moderate capacity.

```text
r = 64
```

```text
4096 → 64 → 4096
```

Much more capacity.

The mathematical constraint is:

[
rank(BA) \leq r
]

So increasing `r` allows LoRA to represent a more complex weight update.

---

## Effect of rank

| Rank | Trainable parameters | Adaptation capacity | Memory   |
| ---- | -------------------: | ------------------- | -------- |
| 2    |             Very low | Low                 | Very low |
| 4    |                  Low | Moderate-low        | Low      |
| 8    |                  Low | Good starting point | Low      |
| 16   |               Higher | Higher              | Higher   |
| 32+  |          Much higher | More expressive     | More     |

There is **no universally best rank**.

You should tune it based on:

* model size
* dataset size
* task complexity
* GPU memory
* evaluation metrics

A reasonable experiment might compare:

```text
r = 4
r = 8
r = 16
r = 32
```

Then select based on validation results and cost.

---

# 3. Code: Understanding rank

```python
import torch
import torch.nn as nn
```

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        r=8
    ):
        super().__init__()

        # Frozen base layer
        self.base = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA A
        self.A = nn.Linear(
            in_features,
            r,
            bias=False
        )

        # LoRA B
        self.B = nn.Linear(
            r,
            out_features,
            bias=False
        )

    def forward(self, x):

        base_output = self.base(x)

        lora_output = self.B(
            self.A(x)
        )

        return base_output + lora_output
```

Test:

```python
layer = LoRALinear(
    in_features=4096,
    out_features=4096,
    r=8
)

print(layer.A.weight.shape)
print(layer.B.weight.shape)
```

Output:

```text
torch.Size([8, 4096])

torch.Size([4096, 8])
```

---

# 4. What is `lora_alpha`?

`alpha` controls the **scaling strength of the LoRA update**.

The common formula is:

[
Y = XW + \frac{\alpha}{r}XBA
]

The scaling factor is:

[
scaling = \frac{\alpha}{r}
]

---

## Example

```text
r = 8
alpha = 16
```

Then:

[
scaling = \frac{16}{8} = 2
]

So:

[
Y = XW + 2XBA
]

The LoRA update is multiplied by `2`.

---

## Why do we need alpha?

Without scaling:

[
Y = XW + XBA
]

The magnitude of the update can change significantly when you change the rank.

For example:

```text
r = 4
```

versus:

```text
r = 64
```

The scale and behavior of the learned update can differ.

`alpha` gives us control over the effective strength of the LoRA branch.

---

## Important distinction

`r` controls:

```text
How much information LoRA can learn
```

`alpha` controls:

```text
How strongly the LoRA update is scaled
```

So:

```text
r
↓
Capacity

alpha
↓
Update scaling
```

---

# 5. Code with `r` and `alpha`

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        alpha=16
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha

        self.scaling = alpha / r

        # Base model
        self.base = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        # Freeze base
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.A = nn.Linear(
            in_features,
            r,
            bias=False
        )

        self.B = nn.Linear(
            r,
            out_features,
            bias=False
        )

    def forward(self, x):

        # Original model output
        base_output = self.base(x)

        # LoRA update
        lora_output = self.B(
            self.A(x)
        )

        # Apply scaling
        return (
            base_output
            +
            self.scaling * lora_output
        )
```

Usage:

```python
layer = LoRALinear(
    in_features=4096,
    out_features=4096,
    r=8,
    alpha=16
)

print(layer.scaling)
```

Output:

```text
2.0
```

---

# 6. What is LoRA dropout?

`lora_dropout` is a regularization technique applied to the **LoRA branch during training**.

Conceptually:

```text
                     ┌───────────────┐
                     │ Frozen W      │
Input ───────────────►               ├──► Output
                     └───────────────┘
                           +
Input
  │
  ▼
Dropout
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

The formula becomes approximately:

[
Y = XW + \frac{\alpha}{r}B A(Dropout(X))
]

---

## Why use dropout?

It helps reduce overfitting, especially when:

* the fine-tuning dataset is small
* the dataset contains repetitive examples
* the model memorizes training samples
* the LoRA adapter is over-specializing

Example:

```python
dropout = nn.Dropout(0.05)
```

Means that during training, some activations in the LoRA path are randomly zeroed according to the dropout probability.

During inference:

```text
Dropout is disabled
```

---

# 7. Complete LoRA implementation

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        alpha=16,
        dropout=0.05
    ):
        super().__init__()

        # Store configuration
        self.r = r
        self.alpha = alpha

        # alpha / r
        self.scaling = alpha / r

        # -----------------------------
        # Base layer
        # -----------------------------

        self.base = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False

        # -----------------------------
        # LoRA matrices
        # -----------------------------

        # A:
        # in_features → r
        self.A = nn.Linear(
            in_features,
            r,
            bias=False
        )

        # B:
        # r → out_features
        self.B = nn.Linear(
            r,
            out_features,
            bias=False
        )

        # Dropout on LoRA path
        self.dropout = nn.Dropout(
            dropout
        )

        # Initialization:
        # Start B at zero so the
        # initial LoRA update is zero.
        nn.init.zeros_(
            self.B.weight
        )

    def forward(self, x):

        # Base output
        base_output = self.base(x)

        # Apply dropout only to
        # LoRA branch
        x_lora = self.dropout(x)

        # LoRA update
        lora_output = self.B(
            self.A(x_lora)
        )

        # Add scaled update
        return (
            base_output
            +
            self.scaling * lora_output
        )
```

Create it:

```python
layer = LoRALinear(
    in_features=4096,
    out_features=4096,
    r=8,
    alpha=16,
    dropout=0.05
)
```

---

# 8. `r`, `alpha`, and dropout together

```text
LoRA Configuration

r = 8
│
└── Low-rank adaptation capacity


alpha = 16
│
└── Scaling strength

alpha / r = 2


dropout = 0.05
│
└── Regularization
```

In code:

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
```

---

# 9. Which layers should you apply LoRA to?

This is extremely important.

You generally do **not automatically apply LoRA to every layer**.

You choose target modules based on:

* model architecture
* task
* available GPU memory
* desired adaptation capacity
* evaluation results

---

# 10. Transformer architecture

A simplified Transformer block:

```text
                 Input
                   │
                   ▼
              LayerNorm
                   │
                   ▼
        ┌────────────────────┐
        │ Multi-Head Attention│
        │                    │
        │ Q Projection       │
        │ K Projection       │
        │ V Projection       │
        │ O Projection       │
        └────────────────────┘
                   │
                   ▼
              Residual
                   │
                   ▼
               LayerNorm
                   │
                   ▼
        ┌────────────────────┐
        │       MLP          │
        │                    │
        │ Up Projection      │
        │ Down Projection    │
        └────────────────────┘
                   │
                   ▼
                Output
```

The major large matrices are:

```text
Attention:
q_proj
k_proj
v_proj
o_proj

MLP:
up_proj
down_proj
gate_proj
```

These are common LoRA targets.

---

# 11. Apply LoRA only to Q and V

A classic approach is:

```text
q_proj  → LoRA
k_proj  → Frozen
v_proj  → LoRA
o_proj  → Frozen
```

Why?

Because:

```text
Q → What information should I look for?
V → What information/content should I carry?
```

Adapting these projections can provide significant task-specific adaptation with relatively few trainable parameters.

In PEFT:

```python
config = LoraConfig(
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

This is a good lightweight starting configuration for architectures that expose modules with these names.

---

# 12. Apply LoRA to all attention projections

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05
)
```

Architecture:

```text
Attention

Input
 ├── q_proj → LoRA ✓
 ├── k_proj → LoRA ✓
 ├── v_proj → LoRA ✓
 └── o_proj → LoRA ✓
```

This provides more adaptation capacity but increases trainable parameters.

---

# 13. Apply LoRA to attention + MLP

For stronger adaptation:

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj"
    ],

    lora_dropout=0.05
)
```

Architecture:

```text
Transformer Block

ATTENTION
q_proj       LoRA ✓
k_proj       LoRA ✓
v_proj       LoRA ✓
o_proj       LoRA ✓

MLP
gate_proj    LoRA ✓
up_proj      LoRA ✓
down_proj    LoRA ✓
```

This usually gives more adaptation capacity, at the cost of:

* more trainable parameters
* more memory
* larger adapter size

---

# 14. How do you know the correct target module names?

**Never blindly copy module names from another model.**

For example:

```text
Llama-style model:

q_proj
k_proj
v_proj
o_proj
```

GPT-2 uses different naming.

Inspect the model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

for name, module in model.named_modules():

    if isinstance(
        module,
        torch.nn.Linear
    ):
        print(name)
```

This helps identify candidate layers.

For models using custom projection layers rather than `nn.Linear`, inspect the architecture accordingly.

---

# 15. GPT-2 example

GPT-2 commonly has combined attention projections such as:

```text
c_attn
```

and:

```text
c_proj
```

So:

```python
config = LoraConfig(
    r=8,
    lora_alpha=16,

    target_modules=[
        "c_attn"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

For broader adaptation, you may target appropriate attention/MLP projection modules after inspecting the specific model.

---

# 16. Llama-style example

For a Llama-style architecture:

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

Or:

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

---

# 17. Full working Hugging Face example

Install:

```bash
pip install torch transformers peft datasets accelerate
```

Imports:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
```

Load model:

```python
MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

GPT-2 does not have a default padding token, so set one:

```python
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
```

Create LoRA configuration:

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

Apply LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check trainable parameters:

```python
model.print_trainable_parameters()
```

You should see something conceptually like:

```text
trainable params: small percentage
all params: total model parameters
trainable%: much less than 100%
```

---

# 18. Why does targeting more layers matter?

Suppose you have a 32-layer model.

### Configuration A

```text
Only q_proj and v_proj

Layer 1     LoRA
Layer 2     LoRA
...
Layer 32    LoRA
```

Trainable parameters:

```text
Low
```

### Configuration B

```text
Q + K + V + O
```

Trainable parameters:

```text
Medium
```

### Configuration C

```text
Attention + MLP
```

Trainable parameters:

```text
Higher
```

Capacity:

```text
Configuration A
       ↓
Configuration B
       ↓
Configuration C

Adaptation capacity increases
```

But:

```text
Trainable parameters increase
GPU memory increases
Training time increases
Adapter size increases
```

---

# 19. How would I choose layers in a real project?

A practical process:

## Step 1: Start small

```python
target_modules = [
    "q_proj",
    "v_proj"
]
```

Evaluate.

---

## Step 2: Measure

Track:

* validation loss
* task accuracy
* F1
* exact match
* hallucination rate
* human evaluation

For RAG systems, also potentially evaluate:

* faithfulness
* answer relevance
* context precision
* context recall

---

## Step 3: Increase capacity if needed

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

Evaluate again.

---

## Step 4: Include MLP if necessary

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj"
]
```

---

## Step 5: Compare experiments

A production experiment table might look like:

| Experiment | Rank | Alpha | Target modules  | Score | Cost   |
| ---------- | ---: | ----: | --------------- | ----: | ------ |
| A          |    8 |    16 | Q, V            |  0.81 | Low    |
| B          |   16 |    32 | Q, K, V, O      |  0.86 | Medium |
| C          |   16 |    32 | Attention + MLP |  0.89 | Higher |

Then choose the best:

```text
Quality / Cost / Latency tradeoff
```

Not simply the model with the highest score.

---

# 20. Recommended starting configurations

These are **starting points for experimentation**, not universal rules.

### Lightweight adaptation

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)
```

---

### Moderate adaptation

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

---

### Strong adaptation

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
        "up_proj",
        "down_proj",
        "gate_proj"
    ]
)
```

Then tune based on actual validation results.

---

# 21. The interview answer

> **LoRA rank `r` determines the dimensionality of the low-rank decomposition and therefore the capacity of the adaptation. A higher rank means more trainable parameters and greater ability to represent complex weight updates.**
>
> **`lora_alpha` is a scaling hyperparameter. The LoRA update is commonly scaled by alpha divided by rank, which controls the effective magnitude of the learned adaptation.**
>
> **`lora_dropout` applies dropout to the LoRA branch during training to regularize the adapter and reduce overfitting.**
>
> **For target layers, I inspect the architecture first and generally start with major attention projections such as query and value. If more adaptation capacity is needed, I expand to all attention projections and then potentially MLP projections. I select the final configuration based on validation quality and compute cost.**

# Memory trick

```text
r
↓
How much can LoRA learn?

alpha
↓
How strongly is LoRA applied?

dropout
↓
How much regularization?

target_modules
↓
Where does LoRA learn?
```

That is the complete practical understanding of the main LoRA configuration parameters.
